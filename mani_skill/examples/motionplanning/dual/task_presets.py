"""Public CLI defaults and report presentation; no acceptance decisions.

Inputs are ordinary CLI strings and saved JSON reports. Presets only supply
missing options for a single public run. Explicit batch manifests and legacy
validation defaults remain unchanged. Summaries never recompute a quality gate.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

PRESET_VERSION = 'tig-dual-arm-task-presets-v1'
TASK_PRESETS = {
    'TwoRobotPickAppleBananaToBaskets-v1': {
        '--approach-recipe': 'compact-4cm',
        '--payload-refinement': 'baseline',
        '--transport-recipe': 'standard'},
    'TwoRobotPickPillToRegions-v1': {
        '--approach-recipe': 'standard-8cm',
        '--payload-refinement': 'fixed-task-v1',
        '--transport-recipe': 'direct-release-v2'},
    'TwoRobotPickFruitsToPlate-v1': {
        '--approach-recipe': 'standard-8cm',
        '--payload-refinement': 'fixed-task-v4',
        '--transport-recipe': 'standard'},
}


def has_option(arguments: list[str], option: str) -> bool:
    return any(item == option or item.startswith(option + '=') for item in arguments)


def with_defaults(arguments: list[str], defaults: Mapping[str, str]) -> list[str]:
    """Prepend missing full options; original explicit arguments retain priority."""
    prefix = [item for flag, value in defaults.items() if not has_option(arguments, flag)
              for item in (flag, value)]
    return prefix + list(arguments)


def generation_arguments(arguments: list[str]) -> list[str]:
    """Apply versioned task recipes only to a public single-episode request."""
    if has_option(arguments, '--manifest'):
        return list(arguments)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', '--task-id', choices=tuple(TASK_PRESETS),
                        default=next(iter(TASK_PRESETS)))
    parser.add_argument('--quality-profile', default='task-default-v1')
    selected, _ = parser.parse_known_args(arguments)
    defaults = {'--quality-profile': 'task-default-v1', '--recording-mode': 'normal'}
    if selected.quality_profile != 'legacy-strict-v1':
        defaults.update(TASK_PRESETS[selected.task])
    return with_defaults(arguments, defaults)


def replay_arguments(arguments: list[str]) -> list[str]:
    """Choose public replay defaults without overriding an explicit strict run."""
    return with_defaults(arguments, {'--replay-mode': 'cross-version', '--recording-mode': 'normal'})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def run_summary(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    """Present recorded independent outcomes; absent evidence stays unknown."""
    results = _mapping(report.get('results'))
    execution = _mapping(results.get('execution'))
    official = _mapping(results.get('official'))
    success = official.get('final_success_before_horizon')
    count = _mapping(report.get('recording')).get('recorded_steps')
    submitted = count > 0 if type(count) is int and count >= 0 else None
    categories = execution.get('failure_categories')
    return {'status': report.get('status', 'unknown'),
        'official_success': success if type(success) is bool else None,
        'physics': _mapping(results.get('physics')).get('status', 'unknown'),
        'quality': _mapping(results.get('quality')).get('status', 'unknown'),
        'recording': _mapping(results.get('recording')).get('status', 'unknown'),
        'phase': execution.get('last_phase') or report.get('stage'),
        'failure_categories': categories if isinstance(categories, list) else [],
        'actions_submitted_including_preparation': submitted,
        'recorded_actions': count if type(count) is int and count >= 0 else None,
        'reason': report.get('error'), 'report': str(report_path)}


def print_run_summary(report_path: Path) -> None:
    """Print a short human summary while detailed receipts stay in report.json."""
    try:
        report = json.loads(report_path.read_text())
        if not isinstance(report, dict):
            raise ValueError('Report is not an object')
    except (OSError, ValueError) as error:
        print(f'Report unavailable: {error}; expected {report_path}', flush=True)
        return
    summary = run_summary(report, report_path)
    flags = {True: 'yes', False: 'no', None: 'unknown'}
    print(f"Result: {summary['status']} | official success: {flags.get(summary['official_success'], 'unknown')}"
          f" | physics: {summary['physics']} | quality: {summary['quality']}"
          f" | recording: {summary['recording']}", flush=True)
    if summary['status'] != 'passed':
        categories = ', '.join(str(value) for value in summary['failure_categories']) or 'unspecified'
        print(f"Stopped at: {summary['phase'] or 'unknown'} | category: {categories}"
              f" | actions submitted (including preparation): {flags[summary['actions_submitted_including_preparation']]}",
              flush=True)
        if summary['reason']:
            # The full reason and traceback remain in the JSON artifact.
            reason = str(summary['reason']).splitlines()[0]
            print('Reason: ' + (reason if len(reason) <= 240 else reason[:237] + '...'), flush=True)
    print(f'Report: {report_path}', flush=True)


def run_public(arguments: list[str], runner: Callable[[list[str]], int]) -> int:
    """Keep normal Python progress in console.log; preserve runner exit status.

    Temporary capture does not create the requested output directory. The
    runner still validates and owns it. Diagnostic/self-test/help retain their
    original output. Native library stderr is not intercepted.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recording-mode', default='normal')
    parser.add_argument('--output-dir', type=Path)
    selected, _ = parser.parse_known_args(arguments)
    if selected.recording_mode != 'normal' or selected.output_dir is None:
        return runner(arguments)
    report_path = selected.output_dir/'report.json'
    print(f'Running; report will be written to {report_path}', flush=True)
    # A preexisting output belongs to the caller and must not receive a log.
    output_existed = selected.output_dir.exists()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as captured:
        try:
            with contextlib.redirect_stdout(captured):
                return runner(arguments)
        finally:
            captured.seek(0)
            if not output_existed and selected.output_dir.is_dir():
                try:
                    with (selected.output_dir/'console.log').open('x') as destination:
                        shutil.copyfileobj(captured, destination)
                except OSError as error:
                    # Presentation must not replace a physical result/exception.
                    print(f'Console log could not be saved: {error}', file=sys.stderr, flush=True)
                print_run_summary(report_path)
            else:
                shutil.copyfileobj(captured, sys.stdout)


def self_test() -> dict[str, Any]:
    """Exercise public defaults, legacy escape, manifest identity and reporting."""
    for task, expected in TASK_PRESETS.items():
        arguments = generation_arguments(['--task', task, '--backend', 'mplib'])
        for flag, value in expected.items():
            assert arguments[arguments.index(flag)+1] == value
    explicit = generation_arguments(['--task-id=TwoRobotPickPillToRegions-v1',
        '--approach-recipe=compact-5cm', '--payload-refinement', 'baseline',
        '--recording-mode=diagnostic', '--transport-recipe', 'standard'])
    assert '--approach-recipe' not in explicit and explicit.count('--payload-refinement') == 1
    assert '--recording-mode' not in explicit and '--transport-recipe' in explicit
    legacy = ['--task', 'TwoRobotPickPillToRegions-v1', '--quality-profile=legacy-strict-v1']
    assert generation_arguments(legacy) == ['--recording-mode', 'normal'] + legacy
    manifest = ['--manifest=data.json', '--output-dir', 'output', '--worker-lifetime=batch']
    original = list(manifest)
    assert generation_arguments(manifest) == original and manifest == original
    assert replay_arguments([]) == ['--replay-mode', 'cross-version', '--recording-mode', 'normal']
    strict = replay_arguments(['--replay-mode=strict', '--recording-mode', 'diagnostic'])
    assert strict == ['--replay-mode=strict', '--recording-mode', 'diagnostic']
    report = {'status': 'failed', 'stage': 'pregrasp', 'recording': {'recorded_steps': 10},
        'error': 'IK rejected', 'results': {'official': {'final_success_before_horizon': False},
        'execution': {'failure_categories': ['planning_failure']},
        'physics': {'status': 'passed'}, 'quality': {'status': 'unknown'},
        'recording': {'status': 'passed'}}}
    before = json.dumps(report, sort_keys=True)
    summary = run_summary(report, Path('/example/report.json'))
    assert summary['official_success'] is False and summary['physics'] == 'passed'
    assert summary['actions_submitted_including_preparation'] is True
    assert summary['failure_categories'] == ['planning_failure']
    assert json.dumps(report, sort_keys=True) == before
    assert run_summary({}, Path('/missing/report.json'))['actions_submitted_including_preparation'] is None
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)/'new'
        def fake_runner(arguments: list[str]) -> int:
            assert not output.exists()
            output.mkdir()
            (output/'report.json').write_text(json.dumps(report))
            print('detailed Python progress')
            return 1
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            assert run_public(['--output-dir', str(output)], fake_runner) == 1
        assert (output/'console.log').read_text() == 'detailed Python progress\n'
        assert 'official success: no' in console.getvalue() and 'Stopped at: pregrasp' in console.getvalue()
        snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
        def rejected_runner(arguments: list[str]) -> int:
            return 2
        with contextlib.redirect_stdout(io.StringIO()):
            assert run_public(['--output-dir', str(output)], rejected_runner) == 2
        assert snapshot == {path.name: path.read_bytes() for path in output.iterdir()}
    return {'status': 'passed', 'preset_version': PRESET_VERSION,
        'task_presets': len(TASK_PRESETS), 'explicit_options_preserved': True,
        'manifest_unchanged': True, 'legacy_recipes_unchanged': True,
        'explicit_strict_replay_preserved': True, 'unknown_results_preserved': True,
        'normal_logs_and_exit_status': True, 'environment_steps': 0, 'stdlib_only': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False))
