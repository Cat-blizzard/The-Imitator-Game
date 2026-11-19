import json
import numpy as np
import pytest

from data_generation_methods.human_to_robot_pipeline.planning_contract import bind
from data_generation_methods.human_to_robot_pipeline.verify_extra_clutter import verify


def fixture(tmp_path, force, timestamps=None, summary=None):
    force = np.asarray(force, dtype=float)
    timestamps = np.arange(len(force))/20 if timestamps is None else np.asarray(timestamps)
    np.savez(tmp_path/"contact.npz", force_N=force, timestamps_seconds=timestamps)
    proposal = {"seed": 42, "split": "train", "added_distractors": []}
    receipt = {
        "length": len(force), "fps": 20, "layout_seed": 42, "split": "train",
        "randomization": {"structured_proposal": proposal},
        "extra_contact": bind(tmp_path/"contact.npz"),
        "max_extra_contact_force_N": float(force.max()) if summary is None else summary,
    }
    (tmp_path/"receipt.json").write_text(json.dumps(receipt))
    (tmp_path/"input.json").write_text(json.dumps({"proposal": proposal}))
    return tmp_path/"receipt.json", tmp_path/"input.json"


def test_force_sidecar_is_verified_but_not_training_approval(tmp_path):
    paths = fixture(tmp_path, [0, .01, 0])
    result = verify(*paths)
    assert result["passed"] and not result["training_admitted"]
    assert not result["full_swept_collision_verified"]


def test_force_threshold_is_not_relaxed(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        verify(*fixture(tmp_path, [0, .5, 0]))


def test_wrong_clock_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        verify(*fixture(tmp_path, [0, 0, 0], timestamps=[0, .1, .2]))


def test_forged_summary_is_recomputed_from_raw(tmp_path):
    with pytest.raises(ValueError, match="summary"):
        verify(*fixture(tmp_path, [0, .2, 0], summary=0))
