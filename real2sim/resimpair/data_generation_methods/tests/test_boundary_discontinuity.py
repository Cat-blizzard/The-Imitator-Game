import cv2
import numpy as np
import pytest

from data_generation_methods.r2h_pipeline.boundary_discontinuity import (
    Config,aligned_residual,tile_energy,temporal_scores,regions,
)


@pytest.fixture
def texture():
    rng=np.random.default_rng(931)
    image=rng.integers(65,135,(160,192,1),dtype=np.uint8)
    return np.repeat(image,3,axis=2)


def sequence_scores(frames):
    residuals=[aligned_residual(a,b)[0] for a,b in zip(frames,frames[1:])]
    scores=temporal_scores([tile_energy(r) for r in residuals])
    return residuals,scores


def test_static_sequence_and_global_exposure_change_are_not_local_candidates(texture):
    frames=[texture.copy() for _ in range(20)]
    frames[-1]=np.clip(texture.astype(np.int16)+25,0,255).astype(np.uint8)
    residuals,scores=sequence_scores(frames)
    # KLT's subpixel fit plus image interpolation may leave a few intensity
    # units even when the synthetic brightness change is exactly global.
    assert np.mean(residuals[-1])<1
    assert not any(s["candidate_tiles"].any() for s in scores if s is not None)


def test_smooth_object_motion_is_not_an_end_teleport(texture):
    frames=[]
    for i in range(28):
        image=texture.copy()
        image[65:75,20+i*2:30+i*2]=[210,30,180]
        frames.append(image)
    residuals,scores=sequence_scores(frames)
    assert not any(regions(r,s) for r,s in zip(residuals[-6:],scores[-6:]) if s is not None)


@pytest.mark.parametrize("colour",[[220,220,220],[20,180,240],[180,230,20]])
def test_small_end_jump_detected_without_colour_identity(texture,colour):
    frames=[]
    for i in range(28):
        image=texture.copy()
        x=24 if i<27 else 142
        image[65:75,x:x+10]=colour
        frames.append(image)
    residuals,scores=sequence_scores(frames)
    last=regions(residuals[-1],scores[-1])
    assert last
    assert residuals[-1].mean()<2.0
    # Review proposals localize both disappearing and appearing regions.
    assert any(r["processing_bbox_xyxy"][0]<40 for r in last)
    assert any(r["processing_bbox_xyxy"][0]>120 for r in last)


def test_global_camera_pan_compensated(texture):
    frames=[cv2.warpAffine(texture,np.array([[1,0,i],[0,1,0]],np.float32),
                          (texture.shape[1],texture.shape[0]),borderMode=cv2.BORDER_REFLECT)
            for i in range(20)]
    residuals,scores=sequence_scores(frames)
    assert np.mean([r.mean() for r in residuals])<1
    assert not any(s["candidate_tiles"].any() for s in scores if s is not None)


def test_missing_temporal_support_remains_unknown():
    assert temporal_scores(np.zeros((3,4,4)))==[None,None,None]
