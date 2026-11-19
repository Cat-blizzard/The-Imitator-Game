"""Bounded public-asset download helpers; no simulator or package installation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "PolicyGen-Household-Asset-Library/1.0 (research; Powered by Poly Haven)"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Guard:
    def __init__(self, root, budget_bytes, reserve_bytes=2**40):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = int(budget_bytes)
        self.reserve = reserve_bytes
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.lock = threading.Lock()

    def charge(self, size):
        with self.lock:
            if self.used + size > self.limit:
                raise RuntimeError("assigned_asset_budget_exceeded")
            if shutil.disk_usage(self.root).free - size < self.reserve:
                raise RuntimeError("1TiB_disk_reserve_reached")
            self.used += size


def fetch_metadata(url, path, *, pause=.6):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            parsed = json.loads(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            write(path.with_name(path.name + ".source.json"), {
                "url": url, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            time.sleep(pause)
            return parsed
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def download(item, path, guard):
    """Range-resumable public GET with size/hash verification and finite retries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item.get("size")
    if path.exists():
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError("Existing final asset has wrong size")
        sha = digest(path)
        if item.get("sha256") and sha != item["sha256"]:
            raise ValueError("Existing final asset has wrong SHA256")
        if item.get("md5") and digest(path, "md5") != item["md5"]:
            raise ValueError("Existing final asset has wrong publisher MD5")
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha,
                "url": item["url"], "publisher_hash_verified": bool(item.get("sha256") or item.get("md5")), "existing": True}
    temporary = path.with_name(path.name + ".part")
    hf_curl = "huggingface.co" in urllib.parse.urlsplit(item["url"]).netloc and expected_size is not None and expected_size > (64 << 20)
    if hf_curl:
        offset = temporary.stat().st_size if temporary.exists() else 0
        guard.charge(max(0, expected_size - offset))
        command = ["curl", "--silent", "--show-error", "--fail", "--location", "--continue-at", "-",
                   "--retry", "3", "--retry-delay", "2", "--connect-timeout", "15", "--max-time", "1800",
                   "--speed-limit", "1024", "--speed-time", "60", "--user-agent", UA,
                   "--output", str(temporary), item["url"]]
        with path.with_name(path.name + ".curl.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log)
            while child.poll() is None:
                if shutil.disk_usage(guard.root).free < guard.reserve:
                    child.terminate();child.wait(timeout=15)
                    raise RuntimeError("1TiB_disk_reserve_reached")
                time.sleep(1)
        if child.returncode != 0:
            raise IOError("Public_HF_curl_failed_exit_" + str(child.returncode))
        if temporary.stat().st_size != expected_size:
            raise IOError("Public_HF_curl_size_mismatch")
    hf_range = False
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            if expected_size is not None and offset == expected_size:
                break
            while expected_size is None or offset < expected_size:
                headers = {"User-Agent": UA}
                if hf_range:
                    # Bounded Range GETs work through the server proxy where
                    # an unbounded large HF GET can wait without response bytes.
                    end = min(offset + (64 << 20) - 1, expected_size - 1)
                    headers["Range"] = f"bytes={offset}-{end}"
                elif offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(item["url"], headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (offset or hf_range) and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("Server did not confirm exact resume offset")
                    with temporary.open("ab" if offset else "wb") as output:
                        while True:
                            block = response.read(4 << 20)
                            if not block:
                                break
                            guard.charge(len(block))
                            output.write(block)
                            offset += len(block)
                if not hf_range:
                    break
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise IOError("Asset transfer ended before declared size")
            break
        except RuntimeError:
            raise
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    actual_sha = digest(temporary)
    if item.get("sha256") and actual_sha != item["sha256"]:
        raise ValueError("Publisher SHA256 mismatch")
    if item.get("md5") and digest(temporary, "md5") != item["md5"]:
        raise ValueError("Publisher MD5 mismatch")
    temporary.replace(path)
    receipt = {"path": str(path), "url": item["url"], "bytes": path.stat().st_size,
               "sha256": actual_sha, "publisher_sha256": item.get("sha256"), "publisher_md5": item.get("md5"),
               "publisher_hash_verified": bool(item.get("sha256") or item.get("md5")),
               "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if hf_curl:
        receipt["transport"] = "system_curl_resumable_public_GET"
    write(path.with_name(path.name + ".source.json"), receipt)
    return receipt
