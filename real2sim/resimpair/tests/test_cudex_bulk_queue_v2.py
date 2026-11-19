"""CPU-only scheduler controls. Fixture outputs are never experiment evidence."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import cudex_bulk_queue_v2 as queue


class BulkQueueTests(unittest.TestCase):
    def make_inputs(self, root, count=5):
        source = root / 'fixture_source.py'; source.write_text('# CPU scheduler fixture\n')
        cfg = root / 'config.yml'; cfg.write_text('seed_num: 256\n')
        gate = root / 'fixture_gate.py'; gate.write_text('# never executed by preparation\n')
        rows = []
        for i in range(count):
            asset = root / 'inputs' / f'asset_{i}.json'
            write(asset, {'fixture': True, 'id': i})
            rows.append({'source_id': f'fixture:{i:04d}', 'content_key': f'key{i:04d}',
                         'asset': binding(asset), 'source_aliases': ['alias:0'] if i == 0 else []})
        index = root / 'index.json'; write(index, {'entries': rows})
        fixed = root / 'fixed.json'; write(fixed, {'entries': rows[:4]})
        return index, cfg, fixed, gate, source

    def make_campaign(self, root, count=5):
        index, cfg, fixed, gate, source = self.make_inputs(root, count)
        campaign = root / 'campaign'
        value = queue.prepare(campaign, index, cfg, cfg, fixed, gate,
                              code_paths=[source], registry=root / 'registry')
        return campaign, value

    def test_complete_index_no_256_cap_and_four_candidate_reuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root, prepared = self.make_campaign(Path(temp), count=300)
            p = read(root / 'protocol.json')
            m = read(root / 'candidate_manifest.json')
            self.assertEqual(prepared['gpu_job_limit'], 304)
            self.assertEqual(len(p['qualification_jobs']), 8)
            self.assertEqual(len(p['bulk_jobs']), 296)
            self.assertEqual(len(m['entries']), 300)
            self.assertEqual(m['source_ids_including_aliases'], 301)
            reused = {r['job']['path'] for r in m['entries']} & {r['path'] for r in p['qualification_jobs']}
            self.assertEqual(len(reused), 4)
            self.assertTrue(all('nonnegative_constraint' in s for s in reused))

    def test_same_recipe_new_directory_cannot_reset_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index, cfg, fixed, gate, source = self.make_inputs(root)
            queue.prepare(root / 'first', index, cfg, cfg, fixed, gate,
                          code_paths=[source], registry=root / 'registry')
            with self.assertRaisesRegex(ValueError, 'existing_recipe_campaign_resume_at:'):
                queue.prepare(root / 'renamed_retry', index, cfg, cfg, fixed, gate,
                              code_paths=[source], registry=root / 'registry')
            self.assertFalse((root / 'renamed_retry').exists())

    def test_frozen_qualification_does_not_change_with_later_index_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index, cfg, fixed, gate, source = self.make_inputs(root, 6)
            data = read(index)
            index.write_text(__import__('json').dumps({'entries': list(reversed(data['entries']))}))
            queue.prepare(root / 'campaign', index, cfg, cfg, fixed, gate,
                          code_paths=[source], registry=root / 'registry')
            p = read(root / 'campaign/protocol.json')
            jobs = [read(queue.resolve(r)) for r in p['qualification_jobs']]
            self.assertEqual([j['source_id'] for j in jobs[::2]], [f'fixture:{i:04d}' for i in range(4)])

    def test_unknown_optimizer_reservation_never_relaunches(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.make_campaign(Path(temp))
            p = read(root / 'protocol.json'); ref = p['qualification_jobs'][0]
            job = queue.resolve(ref)
            write(job.parent / 'reservation.json', {'started_unix': time.time() - 60})
            with patch.object(queue, '_launch_stage') as launch:
                with self.assertRaisesRegex(RuntimeError, 'unknown_reserved_stage'):
                    queue._run_job(root, p, ref, time.time() + 10, {})
                launch.assert_not_called()

    def make_optimizer(self, job):
        out = job.parent
        raw = out / 'raw.fixture'; raw.write_text('not an actual optimizer sample\n')
        components = out / 'components.fixture'; components.write_text('scheduler-only fixture\n')
        write(out / 'optimizer_result.json', {'status': 'optimizer_complete',
            'job': binding(job), 'asset': read(job)['asset'], 'seed_count': 256,
            'nonnegative_semantics_valid': True, 'raw': binding(raw), 'components': binding(components)})

    def test_complete_optimizer_recovers_without_new_gpu_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.make_campaign(Path(temp))
            p = read(root / 'protocol.json'); job = queue.resolve(p['qualification_jobs'][0])
            write(job.parent / 'reservation.json', {'started_unix': time.time() - 60})
            self.make_optimizer(job)
            with patch.object(queue, '_launch_stage') as launch:
                terminal = queue._recover_or_wait(job, 'gpu')
                launch.assert_not_called()
            self.assertEqual(terminal['returncode'], 'recovered_from_complete_output')
            self.assertEqual(terminal['original_process_exit_status'], 'unknown')
            self.assertTrue((job.parent / 'optimizer_terminal.json').is_file())

    def test_pid_reuse_is_not_live_ownership(self):
        identity = {'pid': 100, 'start_ticks': 1, 'boot_id': 'one'}
        with patch.object(queue, 'process_identity', return_value={**identity, 'start_ticks': 2}):
            self.assertFalse(queue.process_alive(identity))
        with patch.object(queue, 'process_identity', return_value={**identity, 'boot_id': 'two'}):
            self.assertFalse(queue.process_alive(identity))

    def test_gate_must_bind_all_current_outputs_and_approve(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.make_campaign(Path(temp))
            p = read(root / 'protocol.json')
            results = []
            for ref in p['qualification_jobs']:
                result = queue.resolve(ref).parent / 'result.json'
                write(result, {'scheduler_fixture': True})
                results.append(binding(result))
            write(root / 'independent_gate.json', {'decision': 'proceed_bulk', 'reviewer': 'fixture',
                'protocol': binding(root / 'protocol.json'), 'qualification_results': results[:-1]})
            with self.assertRaisesRegex(ValueError, 'exact_qualification_outputs'):
                queue.validate_gate(root, p)

    def test_runtime_stage_inherits_gpu_lock_until_worker_finishes(self):
        """Exercise real subprocess/FLOCK inheritance with a CPU-only test worker."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / 'job.json'; write(job, {'fixture': True})
            worker = root / 'fixture_worker.py'
            worker.write_text(
                'import os,time\nfrom pathlib import Path\n'
                'r=Path(os.environ["QUEUE_FIXTURE_ROOT"])\n'
                '(r/"worker_ready").write_text(str(os.getpid()))\n'
                'deadline=time.time()+8\n'
                'while not (r/"release").exists() and time.time()<deadline: time.sleep(.02)\n')
            lock_path = root / 'gpu.lock'
            lock = lock_path.open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = lock.fileno()
            script = ('from pathlib import Path; import sys; '
                      'from resimpair_tasks import cudex_bulk_queue_v2 as q; '
                      'q.WORKER_MODULE="fixture_worker"; '
                      'q.execute_stage(Path(sys.argv[1]), "gpu", 5, (int(sys.argv[2]),))')
            env = dict(os.environ, QUEUE_FIXTURE_ROOT=str(root), CUDA_VISIBLE_DEVICES='',
                       PYTHONPATH=str(root) + os.pathsep + str(queue.ROOT))
            process = subprocess.Popen([sys.executable, '-c', script, str(job), str(fd)],
                cwd=queue.ROOT, env=env, pass_fds=(fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            lock.close()
            try:
                deadline = time.monotonic() + 4
                while not (root / 'worker_ready').exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((root / 'worker_ready').exists())
                with lock_path.open('a+') as competitor:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                (root / 'release').touch()
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
                with lock_path.open('a+') as competitor:
                    fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(read(root / 'optimizer_terminal.json')['returncode'], 0)
            finally:
                (root / 'release').touch()
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
