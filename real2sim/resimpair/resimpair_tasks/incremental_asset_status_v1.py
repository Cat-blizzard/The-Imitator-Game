"""Read-only live CPU queue observation; distinguish running and orphaned jobs."""
import argparse
from collections import Counter
from pathlib import Path
import time

from resimpair_grasp.common import read, write, binding


def matches_process(pid, module, output):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        return module.encode() in args and str(Path(output).resolve()).encode() in args
    except OSError: return False


def observe(root):
    root = Path(root).resolve(); protocol = read(root/'protocol.json')
    supervisor_file = root/'full_cpu_supervisor.json'
    supervisor = read(supervisor_file) if supervisor_file.exists() else {}
    supervisor_alive = matches_process(supervisor.get('pid', -1),
        'resimpair_tasks.incremental_asset_library_v1', root)
    statuses = Counter(); live = []; orphaned = []; finishing = []; seconds = 0.
    for folder in (root/'jobs').iterdir():
        terminal = folder/'terminal.json'
        if terminal.exists():
            value = read(terminal); statuses[value['state']] += 1
            seconds += value.get('elapsed_seconds', 0.); continue
        reservation = folder/'reservation.json'
        if not reservation.exists(): continue
        process_file = folder/'process.json'
        process = read(process_file) if process_file.exists() else {}
        pid = process.get('pid', -1)
        row = {'content_key': folder.name, 'pid': pid, 'reservation': binding(reservation)}
        if matches_process(pid, protocol['converter_module'], folder):
            live.append(row)
        elif supervisor_alive and time.time()-process.get('started_unix', read(reservation).get('started_unix', 0)) <= protocol['worker_timeout_seconds']+30:
            # The worker may have just exited while its owning supervisor is
            # checking native output and atomically writing the terminal.
            finishing.append(row)
        elif terminal.exists():
            value = read(terminal); statuses[value['state']] += 1
            seconds += value.get('elapsed_seconds', 0.)
        else: orphaned.append(row)
    completed = sum(statuses.values())
    return {'schema': 'live_incremental_asset_observation_v1', 'observed_unix': time.time(),
        'root': str(root), 'source_count': protocol['source_count'],
        'unique_geometry_jobs': protocol['unique_geometry_jobs'],
        'supervisor_pid': supervisor.get('pid'), 'supervisor_alive': supervisor_alive,
        'completed_jobs': completed, 'terminal_states': dict(statuses),
        'live_in_flight_count': len(live), 'finalizing_with_live_supervisor_count': len(finishing),
        'orphaned_unknown_count': len(orphaned), 'running': live, 'finalizing': finishing,
        'orphaned': orphaned, 'worker_seconds_completed': seconds,
        'mean_worker_seconds': seconds/completed if completed else None,
        'pending_unique_jobs': protocol['unique_geometry_jobs']-completed-len(live)-len(finishing)-len(orphaned),
        'api_calls': 0, 'gpu_jobs': 0, 'physical_executions': 0,
        'finalizing_is_provisional': True,
        'finalizing_grace_seconds_after_worker_timeout': 30,
        'runtime_and_geometry_checks_not_changed': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); result = observe(a.root)
    if a.output: write(a.output, result)
    print({k:v for k,v in result.items() if k not in ['running', 'finalizing', 'orphaned']})
