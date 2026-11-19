import numpy as np
from PIL import Image

from resimpair_pp2 import grammar as g
from resimpair_pp2 import reference as ref


def test_prompt_names_hand_side_and_order():
    cup, bowl, lemon = {'category': 'cup', 'color': 'red'}, {'category': 'bowl'}, {'category': 'lemon'}
    p = g.program([g.step(cup, 'in', bowl), g.step(lemon, 'next_to', bowl, 'left')], True, 'multi_ordered')
    text = ref.describe_steps(p, [0, 1])
    assert text.startswith('The person uses the right hand to pick up the red cup, puts it into the bowl')
    assert "Then the person uses the left hand to pick up the lemon, sets it down on the table at the person's own left side of the bowl" in text
    prompt = ref.video_prompt({'instruction': g.instruction(p), 'program': p, 'arms': [0, 1]})
    assert 'First put the red cup in the bowl, then put the lemon to the left of the bowl.' in prompt


def test_alignment_detects_shift(tmp_path):
    rng = np.random.default_rng(0)
    base = (rng.random((1024, 1024, 3)) * 255).astype(np.uint8)
    Image.fromarray(base).save(tmp_path / 'a.png')
    Image.fromarray(base).save(tmp_path / 'b.png')
    Image.fromarray(np.roll(base, 5, axis=1)).save(tmp_path / 'c.png')
    assert ref.alignment(tmp_path / 'b.png', tmp_path / 'a.png')['passed']
    shifted = ref.alignment(tmp_path / 'c.png', tmp_path / 'a.png')
    assert not shifted['passed'] and abs(shifted['shift_px'][1]) == 5
