import tempfile
import unittest
from pathlib import Path
from resimpair_grasp.common import write
from resimpair_pickplace.batch_diagnostics import summarize


class BatchDiagnosticsTests(unittest.TestCase):
    def test_failure_pending_and_preflight_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'cohort.json',{'cases':[
                {'id':'rejected','eligible':False}, {'id':'pending','eligible':True},
                {'id':'failed','eligible':True}, {'id':'unstarted','eligible':True}]})
            write(root/'executions/pending/reservation.json',{'reserved':True})
            write(root/'executions/failed/terminal.json',{'physical_attempts':1,'elapsed_seconds':10.})
            write(root/'executions/failed/container_audit.json',{'passed':False})
            write(root/'executions/failed/timing_breakdown.json',{
                'exclusive_times_may_be_summed':True,'rows':[
                    {'phase':'a','operation':'render','exclusive_seconds':3.,'inclusive_seconds':8.},
                    {'phase':'b','operation':'render','exclusive_seconds':2.,'inclusive_seconds':6.}]})
            result=summarize([root]);counts=result['counts']
            self.assertEqual(counts['registered_cases'],4)
            self.assertEqual(counts['physical_attempts'],1)
            self.assertEqual(counts['reserved_without_terminal'],1)
            self.assertEqual(counts['preflight_rejected'],1)
            self.assertEqual(counts['failed'],1)
            self.assertEqual(result['timing']['render']['mean_seconds'],5.)
            with self.assertRaises(ValueError):summarize([root,root])

    def test_absent_timing_and_audit_do_not_become_success_or_zero_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'cohort.json',{'cases':[{'id':'x','eligible':True}]})
            write(root/'executions/x/terminal.json',{'physical_attempts':1,'elapsed_seconds':8.})
            result=summarize([root])
            self.assertEqual(result['counts']['missing_independent_audit'],1)
            self.assertEqual(result['counts']['executions_without_exclusive_timings'],1)
            self.assertNotIn('passed',result['counts'])
            self.assertEqual(result['timing'],{})


if __name__=='__main__':unittest.main()
