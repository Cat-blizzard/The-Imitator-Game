"""Safety boundaries for finite real simulation and role-preserving scene extension."""
import json
from pathlib import Path
import tempfile
import unittest
from data_generation_methods.human_to_robot_pipeline.multi_task_candidate import (
    ledger_reservation, validate_case, check_idle, MAX_ATTEMPTS, launch, worker)
from data_generation_methods.human_to_robot_pipeline.candidate_scene import validate_background
from data_generation_methods.human_to_robot_pipeline.planning_contract import bind
from copy import deepcopy

class MultiTaskCandidateTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        spec=self.root/'spec.json';spec.write_text(json.dumps({'tasks':[{'id':'closed_book_to_open_box','human_windows':[{'id':'train_source','split':'train'},{'id':'validation_source','split':'validation'}]}]}))
        self.case={'schema':'rsh_scene_candidate_v2','scene_inventory':{'static_distractor_count':0,'static_distractors':[]},'case_id':'book_1','task':'closed_book_to_open_box','seed':1,'split':'train','human_spec':bind(spec),'human_window_ids':['train_source'],'background':{}}
    def tearDown(self):self.temp.cleanup()
    def test_output_directory_cannot_reset_campaign_budget(self):
        path=self.root/'case.json';path.write_text(json.dumps(self.case))
        with self.assertRaisesRegex(ValueError,'fixed campaign ledger'):launch(path,self.root/'another_output')
    def test_worker_cannot_bypass_reservation(self):
        path=self.root/'case.json';path.write_text(json.dumps(self.case))
        with self.assertRaisesRegex(RuntimeError,'charged reservation'):worker(path,self.root/'direct_worker')
    def historical_apple_case(self, name):
        # Real preserved bad inputs are regression fixtures, never rewritten.
        base=Path(__file__).resolve().parents[3]/'simulation/cases'
        case=json.loads((base/(name+'.json')).read_text())
        case['schema']='rsh_scene_candidate_v2'
        items=case['proposal']['added_distractors']
        case['scene_inventory']={'static_distractor_count':len(items),'static_distractors':[
            {k:x[k] for k in ('instance_id','asset_uid','category')} for x in items]}
        return case
    def test_real_8006_outer_6057_nested_source_is_rejected(self):
        c=self.historical_apple_case('apple_new_static4_source8006')
        with self.assertRaisesRegex(ValueError,'human source differ'):validate_case(c)
    def test_real_inherited_baseline_obstacle_ids_are_rejected(self):
        c=self.historical_apple_case('apple_new_static4_dev')
        with self.assertRaisesRegex(ValueError,'obstacle identities differ'):validate_case(c)
    def test_real_inherited_zero_clutter_count_is_rejected(self):
        c=self.historical_apple_case('apple_new_static4_dev')
        c['proposal']['planner_requirements']['obstacle_instance_ids']=['source_container','destination_container']+[x['instance_id']for x in c['proposal']['added_distractors']]
        with self.assertRaisesRegex(ValueError,'clutter count differs'):validate_case(c)
    def test_builder_rebuilds_real_stale_source_and_inventory(self):
        from data_generation_methods.human_to_robot_pipeline.scene_case_builder import build_case
        old=self.historical_apple_case('apple_new_static4_source8006')
        new=build_case(case_id='prepared_test_only',task=old['task'],seed=old['seed'],split=old['split'],
            human_window_id='ego_part2_basic_pick_place_8006',background=old['background'],
            proposal=old['proposal'],added_distractors=old['proposal']['added_distractors'])
        self.assertEqual(new['proposal']['human_window_id'],'ego_part2_basic_pick_place_8006')
        self.assertEqual(new['proposal']['human_source_window']['id'],'ego_part2_basic_pick_place_8006')
        self.assertEqual(new['proposal']['factor_design']['clutter_count'],4)
        self.assertEqual(new['scene_inventory']['static_distractor_count'],4)
        self.assertEqual(new['execution_status'],'prepared_not_executed')
        validate_case(new)
    def test_body_category_count_and_identity_bindings(self):
        c=self.historical_apple_case('apple_new_static4_dev_original_background')
        validate_case(c)
        for key,value in [('instance_id','renamed'),('category','wrong'),('asset_uid','different')]:
            bad=deepcopy(c);bad['scene_inventory']['static_distractors'][0][key]=value
            with self.assertRaisesRegex(ValueError,'IDs/categories/count'):validate_case(bad)
        c['scene_inventory']['static_distractor_count']=0
        with self.assertRaisesRegex(ValueError,'IDs/categories/count'):validate_case(c)
    def test_cross_split_human_derivative_rejected(self):
        validate_case(self.case)
        self.case['human_window_ids']=['validation_source']
        with self.assertRaisesRegex(ValueError,'frozen data split'):validate_case(self.case)
    def test_native_expert_does_not_silently_accept_unplanned_clutter(self):
        self.case['added_distractors']=[{'category':'hammer'}]
        with self.assertRaisesRegex(ValueError,'not qualified'):validate_case(self.case)
    def test_six_cumulative_submissions_count_unknowns(self):
        for i in range(MAX_ATTEMPTS):
            self.case['case_id']=str(i);ledger_reservation(self.root,self.case,{})
        self.case['case_id']='seventh'
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):ledger_reservation(self.root,self.case,{})
    def test_touched_identity_not_retried(self):
        ledger_reservation(self.root,self.case,{})
        with self.assertRaises(FileExistsError):ledger_reservation(self.root,self.case,{})
    def test_gpu_guard_catches_other_owner(self):
        check_idle({'compute_owners':[],'memory_MiB':21,'utilization':0})
        with self.assertRaises(RuntimeError):check_idle({'compute_owners':['someone'],'memory_MiB':21,'utilization':0})
    def test_texture_hash_and_provenance_required(self):
        image=self.root/'asset.bin';image.write_bytes(b'fixed')
        item={**bind(image),'asset_id':'example','license':'CC0','source':'explicit-source'}
        validate_background({'wall':item})
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):validate_background({'wall':item})
        with self.assertRaises(ValueError):validate_background({'new_collider':item})

if __name__=='__main__':unittest.main()
