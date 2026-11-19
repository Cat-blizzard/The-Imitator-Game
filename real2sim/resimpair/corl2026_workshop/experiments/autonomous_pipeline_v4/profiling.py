"""Read-only duration/failure audit of real campaigns, with explicit clock scope.

No filesystem mtime is used as an execution timestamp. Transport timers,
worker timers and cross-stage wall-clock intervals are separate quantities.
Source intake, shared anchors, first runs and retrospective probes are separate
cohorts. A provider response is never interpreted as a qualified data pair.
"""
from collections import defaultdict
import csv
import io
from pathlib import Path
import statistics

from .core import bind, checked, implementation, read, require, write, write_bytes

CATEGORIES = {"resimpair_source_request_v4": "source_extract",
              "resimpair_initial_request_v4": "anchor_generate",
              "resimpair_anchor_review_request_v4": "anchor_review",
              "resimpair_video_request_v4": "video_submit",
              "resimpair_video_review_request_v4": "video_review"}


def summary(values):
    values = [v for v in values if v is not None]
    return {"n_timed": len(values), "min_seconds": min(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
            "max_seconds": max(values) if values else None,
            "sum_seconds": sum(values) if values else None}


def cohort(request, root):
    try:
        first = request.relative_to(root).parts[0]
    except ValueError:
        return "external_registered_request"
    return {"sources": "source_intake", "anchors": "shared_anchor", "runs": "first_generation",
            "review_audits": "retrospective_review", "diagnostics": "interface_probe"}.get(first, "other")


def calls(root):
    budget = read(root / "budget.json")
    rows = []
    for marker in sorted(Path(budget["ledger_path"]).glob("stages/*/call_started.json")):
        started = read(marker)
        request_path = checked(started["request"])
        request = read(request_path)
        stage = marker.parent
        completed, failed = (stage / "result.json").exists(), (stage / "failure.json").exists()
        seconds, metric, clock_evidence = None, "unknown", None
        # The envelope timer covers preparation/normalization too. Prefer it
        # on future runs; old HTTP-only durations remain labelled as such.
        spans = [p for p in stage.glob("timings/*/finished.json") if read(p)["stage"] == "api_" + started["kind"]]
        if spans:
            require(len(spans) == 1, "duplicate_counted_api_operation_timings")
            clock_evidence = spans[0]
            seconds, metric = read(spans[0])["elapsed_seconds"], "operation_wall"
        elif failed:
            clock_evidence = stage / "failure.json"
            end = read(clock_evidence).get("finished_unix")
            if end is not None:
                seconds, metric = end - started["started_unix"], "failed_operation_wall"
        else:
            for filename in ("transport_receipt.json", "stream_timing.json"):
                if (stage / filename).exists():
                    clock_evidence = stage / filename
                    seconds, metric = read(clock_evidence)["elapsed_seconds"], "http_transport"
                    break
        require(seconds is None or seconds >= 0, "negative_clock_interval")
        if completed and started["kind"] == "video":
            terminal = stage / "video_terminal.json"
            state = read(terminal)["status"] if terminal.exists() else "in_flight"
        else:
            state = "returned" if completed else "failed" if failed else "unknown"
        rows.append({"campaign": root.name, "id": stage.name, "category": CATEGORIES.get(request.get("schema"), "api_probe"),
                     "cohort": cohort(request_path, root), "kind": started["kind"], "request": started["request"],
                     "input_image_count": len(request.get("images", [])), "status": state,
                     "duration_seconds": seconds, "duration_scope": metric,
                     "started_unix": started["started_unix"], "started_evidence": bind(marker),
                     "timing_evidence": bind(clock_evidence) if clock_evidence else None,
                     "http_status": read(stage / "failure.json").get("http_status") if failed else None})
    return rows


def executions(root):
    rows = []
    for path in sorted((root / "executions").glob("*/*/terminal.json")):
        terminal = read(path)
        worker = terminal.get("result") or {}
        failure_path = path.parent / "planning_failure.json"
        failure = read(failure_path) if failure_path.exists() else worker.get("planning_failure")
        success = bool(worker.get("bundle")) and not failure and worker.get("status") == "planned_and_executed"
        rows.append({"campaign": root.name, "case": path.parent.parent.name, "attempt": path.parent.name,
                     "status": "completed_export" if success else "failed_attempt",
                     "duration_seconds": terminal.get("elapsed_seconds"), "duration_scope": "worker_wall_including_setup_planning_execution_render_export",
                     "frames": worker.get("frames"), "failure_type": failure.get("type") if failure else None,
                     "raw_supervisor_status": terminal.get("status"), "timed_out": terminal.get("timed_out"),
                     "status_disagreement": (terminal.get("status") == "executed") != success,
                     "evidence": bind(path)})
    return rows


def runs(root, call_rows, execution_rows):
    rows = []
    for request_path in sorted((root / "runs").glob("*/*/video_request.json")):
        request = read(request_path)
        video = next((r for r in call_rows if r["request"]["path"] == str(request_path)), None)
        reviews = [r for r in call_rows if r["request"]["path"] == str(request_path.parent / "review_request.json")]
        if video is None:
            continue
        review = min(reviews, key=lambda r: r["started_unix"]) if reviews else None
        contract = read(checked(request["contract"]))
        execution_path = Path(contract["execution"]["path"])
        simulation = next((r for r in execution_rows if r["case"] == execution_path.parent.parent.name
                           and r["attempt"] == execution_path.parent.name), None)
        end_path = request_path.parent / "outcome.json"
        outcome = read(end_path) if end_path.exists() else {}
        anchor_dir = Path(request["first_frame"]["path"]).parent
        anchor = next((r for r in call_rows if r["request"]["path"] == str(anchor_dir / "image_request.json")), None)
        anchor_review = next((r for r in call_rows if r["request"]["path"] == str(anchor_dir / "review_request.json")), None)
        elapsed = review["started_unix"] - video["started_unix"] if review else None
        require(elapsed is None or elapsed >= 0, "review_precedes_video_submission")
        gate = outcome.get("diagnosis") or {}
        gate = gate.get("gate", {})
        rows.append({"campaign": root.name, "case": request_path.parent.parent.name,
                     "release": request_path.parent.name, "video_submission_start": video["started_unix"],
                     "video_until_review_start_seconds": elapsed,
                     "video_interval_scope": "submission-start to first review-call-start; includes queue/generation, polling, download, local frame preparation and any orchestration gaps; not pure provider compute",
                     "review_transport_seconds": review["duration_seconds"] if review else None,
                     "review_input_images": review["input_image_count"] if review else None,
                     "simulation_worker_seconds": simulation["duration_seconds"] if simulation else None,
                     "anchor_id": anchor_dir.name, "anchor_generation_seconds": anchor["duration_seconds"] if anchor else None,
                     "anchor_review_seconds": anchor_review["duration_seconds"] if anchor_review else None,
                     "original_outcome": outcome.get("status", "unknown"),
                     "failed_checks": gate.get("failed_checks", []), "unknown_checks": gate.get("unknown_checks", []),
                     "training_admitted": outcome.get("training_admitted", False),
                     "evidence": {"video_started": video["started_evidence"],
                                  "review_started": review["started_evidence"] if review else None,
                                  "outcome": bind(end_path) if end_path.exists() else None,
                                  "contract": request["contract"]}})
    seen = set()
    for row in sorted(rows, key=lambda r: r["video_submission_start"]):
        key = row["anchor_id"]
        row["shared_anchor_reused"] = key in seen
        row["anchor_video_consumers"] = sum(r["anchor_id"] == key for r in rows)
        seen.add(key)
    return rows


def span_rows(root):
    """Include interrupted spans; never sum parents and children as wall time."""
    result = []
    for path in sorted(root.glob("**/timings/*/started.json")):
        finished = path.parent / "finished.json"
        row = read(finished if finished.exists() else path)
        result.append({**row, "status": row.get("status", "unfinished_unknown"),
                       "elapsed_seconds": row.get("elapsed_seconds"),
                       "evidence": bind(finished if finished.exists() else path)})
    return result


def audit(roots, output):
    roots, output = [Path(r).resolve() for r in roots], Path(output).resolve()
    require(len(roots) == len(set(roots)), "duplicate_campaign_roots")
    require(all(output != r and r not in output.parents for r in roots), "profiling_output_must_be_outside_retained_campaigns")
    ledgers = [str(Path(read(r / "budget.json")["ledger_path"]).resolve()) for r in roots]
    require(len(ledgers) == len(set(ledgers)), "shared_ledger_would_double_count_calls")
    all_calls, all_executions, all_runs, all_spans = [], [], [], []
    for root in roots:
        c, e = calls(root), executions(root)
        all_calls += c
        all_executions += e
        all_runs += runs(root, c, e)
        all_spans += span_rows(root)
    groups = defaultdict(list)
    for row in all_calls:
        key = tuple(row[k] for k in ("campaign", "category", "cohort", "status", "duration_scope", "input_image_count"))
        groups[key].append(row)
    aggregates = [{**dict(zip(("campaign", "category", "cohort", "status", "duration_scope", "input_image_count"), key)),
                   "n_calls": len(values), **summary([r["duration_seconds"] for r in values])}
                  for key, values in sorted(groups.items())]
    result = {"schema": "resimpair_runtime_profile_v1", "implementation": implementation(), "campaigns": [str(r) for r in roots],
              "calls": all_calls, "executions": all_executions, "runs": all_runs,
              "call_groups": aggregates, "instrumented_spans": all_spans,
              "unknown_components": ["historical asset acquisition and adapter/agent development labor",
                  "historical source window search and selection", "historical compilation and local frame decoding",
                  "historical pure IK versus simulation/render split", "provider queue versus inference split",
                  "historical exact video-ready and download timestamps", "actual currency debits"],
              "rules": {"mtime_used_as_timer": False, "parent_child_durations_summed": False,
                        "retrospective_reviews_mixed_with_first_runs": False,
                        "schema_versions_or_frame_budgets_pooled_for_quality_rate": False,
                        "no_random_task_success_probability_inferred": True},
              "new_paid_calls": 0, "new_simulations": 0,
              "currency_per_valid_pair": None, "scope": "retrospective logs, not a fresh randomized benchmark; original outcome labels are not revised anatomical/identity ground truth"}
    write(output / "profile.json", result)
    _csv(output / "stage_calls.csv", all_calls, ["campaign", "id", "category", "cohort", "input_image_count", "status", "duration_seconds", "duration_scope", "http_status"])
    _csv(output / "task_runs.csv", all_runs, ["campaign", "case", "release", "simulation_worker_seconds", "anchor_generation_seconds", "anchor_review_seconds", "shared_anchor_reused", "video_until_review_start_seconds", "review_transport_seconds", "review_input_images", "original_outcome"])
    _csv(output / "simulation_attempts.csv", all_executions, ["campaign", "case", "attempt", "status", "duration_seconds", "frames", "failure_type", "status_disagreement"])
    write_summary(result, output / "summary.md")
    return {"profile": str(output / "profile.json"), "paid_calls_audited": len(all_calls),
            "physical_attempts_audited": len(all_executions), "human_video_runs": len(all_runs),
            "instrumented_spans": len(all_spans), "new_paid_calls": 0}


def _csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode("utf-8-sig"))


def write_summary(result, path):
    def seconds(value):
        return "unknown" if value is None else f"{value:.1f}"
    lines = ["# 实际运行耗时审计", "", "旧记录只读；时间未知不填零，不用文件修改时间。API 返回成功不等于内容合格。", "",
             "## 各任务已记录的阶段（秒）", "",
             "| 任务 | 仿真进程 | 首图生成 | 首图审核 | 首图已复用 | 视频提交→审核开始 | 视频审核 HTTP | 原协议结论 |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for row in sorted(result["runs"], key=lambda r: r["video_submission_start"]):
        lines.append("| " + " | ".join([row["case"], seconds(row["simulation_worker_seconds"]),
                     seconds(row["anchor_generation_seconds"]), seconds(row["anchor_review_seconds"]),
                     "是" if row["shared_anchor_reused"] else "否", seconds(row["video_until_review_start_seconds"]),
                     seconds(row["review_transport_seconds"]), row["original_outcome"]]) + " |")
    lines += ["", "首图时间描述共享资产的一次实际生成；复用行不代表再次花费。视频区间含排队/生成/轮询/下载/抽帧/编排空隙，不能写成纯模型推理时间。历史通过标签按当时协议保留，未经新版人体和对象规则重新放行。", "",
              "## API 分组（秒）", "",
              "| Campaign | 阶段 | 队列 | 状态 | 图片数 | 调用数/有计时数 | 最短/中位/最长 | 时间口径 |",
              "|---|---|---|---|---:|---|---|---|"]
    for row in result["call_groups"]:
        lines.append("| " + " | ".join([row["campaign"], row["category"], row["cohort"], row["status"],
            str(row["input_image_count"]), f"{row['n_calls']}/{row['n_timed']}",
            "/".join(seconds(row[key]) for key in ("min_seconds", "median_seconds", "max_seconds")), row["duration_scope"]]) + " |")
    lines += ["", "## 仿真：失败不能从分母中删除", "",
              "| 任务/尝试 | 结果 | 进程秒数 | 记录帧数 | 原 supervisor 标签冲突 |", "|---|---|---:|---:|---|"]
    for row in result["executions"]:
        lines.append(f"| {row['case']}/{row['attempt']} | {row['status']} | {seconds(row['duration_seconds'])} | {row['frames']} | {row['status_disagreement']} |")
    lines += ["", "## 缺失和统计边界", ""] + ["- " + item for item in result["unknown_components"]]
    lines += ["", "样本是研发过程中选择的任务，不能据此估计任意随机任务成功率。少量样本只报告 n、最短、中位和最长，不伪造稳定 P95。重审和接口试探单列；父子 span 不相加。实际费用未知，正式合格配对仍不能由 VLM 通过数替代。", ""]
    write_bytes(path, "\n".join(lines).encode())


def plot(profile_path, output_path):
    """Standalone comparison of disjoint recorded stages, not E2E latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = read(profile_path)
    rows = sorted(data["runs"], key=lambda r: r["video_submission_start"])
    require(rows, "no_completed_video_paths_to_plot")
    components = [
        ("Simulation worker", "simulation_worker_seconds", "#40798c"),
        ("New shared anchor + review", None, "#dca34b"),
        ("Video submit to review start*", "video_until_review_start_seconds", "#7e75b3"),
        ("Video review HTTP", "review_transport_seconds", "#ca6c5e"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    left = [0.] * len(rows)
    for label, key, color in components:
        values = []
        for row in rows:
            if key is None:
                require(row["shared_anchor_reused"] or (row["anchor_generation_seconds"] is not None
                        and row["anchor_review_seconds"] is not None), "plot_anchor_duration_unknown")
                v = 0. if row["shared_anchor_reused"] else row["anchor_generation_seconds"] + row["anchor_review_seconds"]
            else:
                v = row[key]
            require(v is not None, "plot_requires_known_durations_not_zero_filled_unknowns")
            values.append(v)
        ax.barh(range(len(rows)), values, left=left, label=label, color=color, height=.62)
        left = [a + b for a, b in zip(left, values)]
    labels = [r["case"].replace("mechanism_workbench_", "Workbench ").replace("scene_", "Scene ").replace("__", " / ") for r in rows]
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of recorded stages (seconds); NOT a measured end-to-end latency")
    ax.set_title("Six retained human-video runs: where the recorded time went", loc="left", pad=18, fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    for i, value in enumerate(left):
        ax.text(value + 5, i, f"{value/60:.1f} min", va="center", fontsize=10)
    ax.set_xlim(0, max(left) * 1.13)
    fig.text(.015, .015, "*Includes queue/generation, polling, download and local preparation. Source intake, adapter development and unlogged gaps excluded.\nAnchor cost counted once per shared scene. Original review results are not a guarantee of valid training pairs.", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, .12, 1, 1])
    output_path = Path(output_path)
    require(not output_path.exists(), "profile_plot_output_exists")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)
