"""Create a local, frame-exact viewing bundle; never a training release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.common import file_sha256, probe_video, write_json


def link(source, destination):
    source = Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError("Viewing link already points to a different source")
        return
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.root / "input_manifest.json").read_text())
    pairs = json.loads(args.pair_manifest.read_text())["pairs"]
    target = args.root / "review_bundle"
    windows = target / "source_windows"
    windows.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, job in enumerate(inputs["jobs"], 1):
        source = job["source_window"]
        src = source["video"]
        if file_sha256(src["path"]) != src["sha256"]:
            raise ValueError("Original real-human source changed")
        start, end = source["window_frames"]
        key = f"{src['sha256'][:16]}_{start}_{end}"
        clip = windows / f"{key}.mp4"
        if not clip.exists():
            subprocess.run([
                "ffmpeg", "-v", "error", "-n", "-i", src["path"],
                "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-pix_fmt", "yuv420p", str(clip),
            ], check=True, timeout=300)
        media = probe_video(clip)
        if media["frames"] != end - start or abs(media["fps"] - source["fps"]) > 1e-6:
            raise ValueError("Exported human task window changed its frame count or fps")
        matches = [pair for pair in pairs if pair["pair_id"].startswith(job["job_id"] + "__")]
        if len(matches) != 1:
            raise ValueError("Viewing case must have one explicit cross-demo pair")
        pair = matches[0]
        query = pair["query"]["episode_row"]
        query_video = query.get("reference_video") or query["video"]
        if isinstance(query_video, dict):
            query_video = query_video["path"]
        directory = target / f"{number:02d}_{job['job_id']}"
        directory.mkdir(exist_ok=True)
        assets = {
            "01_real_human_task_window.mp4": clip,
            "02_sim_generation_donor.mp4": job["robot"]["video"]["path"],
            "03_synthetic_human_candidate.mp4": pair["reference"]["video"]["path"],
            "04_independent_robot_query.mp4": query_video,
        }
        for name, path in assets.items():
            link(path, directory / name)
        record = {
            "job_id": job["job_id"], "pair_id": pair["pair_id"],
            "real_human_original": src, "original_window_frames": [start, end],
            "interval": "start_inclusive_end_exclusive",
            "real_human_window_preview": media,
            "pair_manifest": {"path": str(args.pair_manifest.resolve()),
                              "sha256": file_sha256(args.pair_manifest)},
            "videos": {name: {"path": str((directory / name).resolve()),
                              "sha256": file_sha256(directory / name)}
                       for name in assets},
            "scope": "task-only viewing; independent clocks; no framewise correspondence",
            "symlinks_require_original_local_files": True,
            "quality_release_approved": False, "training_admitted": False,
        }
        write_json(directory / "provenance.json", record)
        rows.append({"directory": str(directory), "job_id": job["job_id"],
                     "pair_id": pair["pair_id"]})
    write_json(target / "bundle_manifest.json", {
        "cases": rows, "unique_real_human_windows": len(set(
            (j["source_window"]["video"]["sha256"], tuple(j["source_window"]["window_frames"]))
            for j in inputs["jobs"])),
        "purpose": "inspection_only_not_training_release",
    })
    (target / "README.md").write_text(
        "# 本轮 real → sim → synthetic-human 查看目录\n\n"
        "4个案例、3个独立真人任务窗口。每个案例按01–04观看：\n\n"
        "1. 真实人类的准确任务片段，按原始帧号裁切；书本只取190–260帧。\n"
        "2. 用于生成初图的已有机器人仿真执行A。\n"
        "3. 由初图＋任务文字生成的人类参考候选。\n"
        "4. 显式配对的另一次机器人执行B，训练时应使用B自己的观测和动作。\n\n"
        "四列不是逐帧对齐，不共享时间长度或动作标签；也不是16个独立样本。"
        "合成人类视频没有真实3D手腕标签。柜门案例存在额外手及柜内物体不可确认问题。"
        "所有案例仍为inspection-only，未正式发布或用于训练。\n\n"
        "目录使用本机软链接，不能单独搬走后假定视频仍可访问。每个案例的provenance.json"
        "记录来源、窗口、文件hash和明确配对清单。原始生成失败和审查结果保留在上一级jobs中。\n",
        encoding="utf-8")
    print(json.dumps({"review_bundle": str(target), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
