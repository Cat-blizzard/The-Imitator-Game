"""Deterministic evidence and video delivery; no task-specific hand-picked frames."""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import os
from pathlib import Path
import time

from .core import bind, read, replace_state
from .intake import cases, sources
from .budget import Budget


def href(path, base):
    return html.escape(os.path.relpath(path, base), quote=True)


def report(root):
    root = Path(root)
    gallery = root / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    summaries = []
    counts = Counter()
    failures = Counter()
    references = defaultdict(list)
    sections = []
    for case in cases(root):
        attempts = []
        for path in sorted((root / "executions" / case["id"]).glob("*/terminal.json")):
            record = read(path)
            attempts.append({"terminal": bind(path), "status": record["status"],
                             "frames": (record.get("result") or {}).get("frames", 0)})
            counts["simulation_attempts_including_setup_failures"] += 1
            counts["native_recorded_frames"] += attempts[-1]["frames"]
            if attempts[-1]["frames"] == 0:
                counts["setup_attempts_without_recorded_execution"] += 1
        handoff_path = root / "planner_handoffs" / (case["id"] + ".json")
        handoff = read(handoff_path) if handoff_path.exists() else None
        compiled = None
        if handoff:
            counts["completed_planning_exports"] += 1
            contract_path = root / "compiled" / case["id"] / handoff["attempt"] / "contract.json"
            if not contract_path.exists():
                versions = sorted((root / "compiled_by_release").glob(f'*/{case["id"]}/{handoff["attempt"]}/contract.json'))
                if versions:
                    contract_path = versions[-1]
            if contract_path.exists():
                compiled = read(contract_path)
                counts["compiled_physically_qualified_executions"] += 1
        runs = []
        for outcome_path in sorted((root / "runs" / case["id"]).glob("*/outcome.json")):
            result = read(outcome_path)
            video_path = outcome_path.parent / "human.mp4"
            counts["terminal_pipeline_runs"] += 1
            counts[result["status"]] += 1
            if video_path.exists():
                counts["generated_human_videos"] += 1
                references[case["semantic_key"]].append({"case": case["id"], "outcome": bind(outcome_path),
                                                       "video": bind(video_path), "quality_status": result["status"]})
            gate = (result.get("diagnosis") or {}).get("gate", {})
            for name in gate.get("failed_checks", []):
                failures["no:" + name] += 1
            for name in gate.get("unknown_checks", []):
                failures["unknown:" + name] += 1
            runs.append({"outcome": bind(outcome_path), "status": result["status"],
                         "video": bind(video_path) if video_path.exists() else None})
        summary = {"case": case["id"], "scene": case["scene"], "task": case["task"], "partition": case["partition"],
                   "semantic_key": case["semantic_key"], "attempts": attempts, "handoff": bind(handoff_path) if handoff else None,
                   "compiled": bind(contract_path) if compiled else None, "human_runs": runs}
        summaries.append(summary)
        videos = []
        if handoff:
            bundle = read(handoff["execution_bundle"]["path"])
            for name, label in [("robot_side", "机器人一侧 · 同步记录"), ("opposite_high", "桌面对侧高机位 · 同步记录")]:
                c = bundle["cameras"][name]
                videos.append(f'<div><p>{label}</p><video controls preload="metadata" poster="{href(c["initial_image"]["path"], gallery)}" src="{href(c["video"]["path"], gallery)}"></video></div>')
        for run in runs:
            if run["video"]:
                videos.append(f'<div><p>合成人类参考 · {html.escape(run["status"])}</p><video controls preload="metadata" src="{href(run["video"]["path"], gallery)}"></video><a href="{href(run["outcome"]["path"], gallery)}">自动审核与失败记录</a></div>')
            else:
                videos.append(f'<div><p>流程终态：{html.escape(run["status"])}</p><a href="{href(run["outcome"]["path"], gallery)}">自动诊断</a></div>')
        sections.append(f'<section><h2>{html.escape(case["id"])}</h2><p>{html.escape(case["partition"])} · {len(attempts)} 次仿真尝试</p><div class="videos">{"".join(videos)}</div></section>')
    coverage = [{"case": s["case"], "semantic_key": s["semantic_key"], "candidate_references": references[s["semantic_key"]],
                 "scope": "task-semantic reference reuse across permitted nuisance/layout changes, not pixel or temporal alignment",
                 "training_admitted": False} for s in summaries]
    scene_groups = defaultdict(list)
    for s in summaries:
        if s["handoff"]:
            h = read(s["handoff"]["path"])
            b = read(h["execution_bundle"]["path"])
            scene_groups[b["scene"]["id"]].append({"case": s["case"], "initial_objects": b["initial_object_state_sha256"],
                                                     "initial_rgb": b["cameras"]["opposite_high"]["initial_image"]["sha256"]})
    shared = {name: {"cases": rows, "same_initial_object_state": len({r["initial_objects"] for r in rows}) == 1,
                     "same_initial_rgb": len({r["initial_rgb"] for r in rows}) == 1} for name, rows in scene_groups.items()}
    counts["source_model_contracts"] = len(list((root / "sources").glob("*/source_contract.json")))
    counts["distinct_declared_task_families"] = len({read(s["task"]["path"])["family"] for s in summaries})
    counts["synchronized_robot_videos"] = 2 * counts["completed_planning_exports"]
    counts["formal_training_admitted_pairs"] = 0
    result = {"schema": "resimpair_autonomy_report_v4", "updated_unix": time.time(), "counts": dict(counts),
              "resources": Budget(root / "budget.json").usage(), "cases": summaries, "same_scene_comparisons": shared,
              "reference_coverage": coverage, "failure_check_counts": dict(failures),
              "source_contracts": [bind(p) for p in sorted((root / "sources").glob("*/source_contract.json"))],
              "manual_post_planner_observation_files": 0, "automatic_semantic_retries": 0,
              "caveats": ["Autonomy and visual quality are counted separately.",
                          "Model observations are frame-bound model evidence, not independent human ground truth.",
                          "Physical qualification is native sampled contact/state validation, not strict all-path certification.",
                          "New layouts share source sessions and retain their original source data split.",
                          "No policy training, full human 3D labels, or all-frame quality release."]}
    replace_state(root / "report.json", result)
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>ReSimPair v4 · 统一流程多任务验证</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#101720;color:#ecf2f5}a{color:#8fd3ff}section{border-top:1px solid #46515e;margin-top:25px;padding-top:12px}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px}video{width:100%;background:#000;border-radius:8px}p{line-height:1.6;color:#c5d1da}</style>'
                '<h1>ReSimPair v4：一个场景，多种任务</h1><p>来源由固定模型协议提取；规划导出后自动编译、生成、审核。生成失败保留为终态，不手填审核或逐任务补 prompt。两路机器人视频来自同次执行同步采集。</p>'
                f'<p>已编译物理执行 {counts["compiled_physically_qualified_executions"]} · 已生成人类视频 {counts["generated_human_videos"]} · 采样视觉通过 {counts["sampled_visual_pass"]}。视觉判断是模型在采样帧上的证据，不是逐帧或三维物理保证。</p>'
                '<p><a href="../report.json">完整运行、资源、参考复用和失败证据</a></p>' + ''.join(sections) + '</html>')
    temporary = gallery / f"index.{os.getpid()}.tmp"
    temporary.write_text(document)
    os.replace(temporary, gallery / "index.html")
    return {"report": str(root / "report.json"), "gallery": str(gallery / "index.html"), "counts": dict(counts),
            "resources": result["resources"]}
