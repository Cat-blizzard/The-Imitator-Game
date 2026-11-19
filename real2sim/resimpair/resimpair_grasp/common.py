from pathlib import Path
import hashlib
import json
import os

from resimpair_runtime import ROOT, configure
configure()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    """New evidence only; retries may read identical evidence, never replace it."""
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Refusing to overwrite evidence: " + str(path))
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    os.link(temporary, path)
    temporary.unlink()


_MANI_SKILL_LINK = ROOT / "PolicyGen" / "mani_skill"   # Imitator 2.0: a link to the repository's ManiSkill fork


def binding(path):
    path = Path(path).resolve()
    if _MANI_SKILL_LINK.is_symlink():                   # record fork files under their ReSimPair-relative name
        target = _MANI_SKILL_LINK.resolve()
        if path.is_relative_to(target):
            return {"path": str(Path("PolicyGen/mani_skill") / path.relative_to(target)), "sha256": sha(path)}
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def resolve(value):
    path = Path(value["path"])
    path = path if path.is_absolute() else ROOT / path
    if sha(path) != value["sha256"]:
        raise ValueError("Asset/evidence hash changed: " + str(path))
    return path


def code_bindings():
    paths = list((ROOT / "resimpair_grasp").glob("*.py"))
    paths += list((ROOT / "third_party/grasp_sampler/src/grasp_sampler").glob("*.py"))
    return {str(p.relative_to(ROOT)): binding(p) for p in sorted(paths)}
