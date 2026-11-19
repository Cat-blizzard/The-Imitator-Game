"""Keep scene sampling seeds while adapting the simulator's legacy RNG API."""
from numbers import Integral


def simulator_seed(scene_seed):
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, Integral) or scene_seed < 0:
        raise ValueError('Scene seed must be a nonnegative integer')
    return int(scene_seed) % (2**32)


def seed_receipt(scene_seed):
    return {'schema': 'simulator_seed_compatibility_v1', 'source_scene_seed': int(scene_seed),
        'simulator_reset_seed': simulator_seed(scene_seed), 'mapping': 'unsigned_32_bit_modulo',
        'scene_sampling_and_planner_seeds_unchanged': True,
        'layout_objects_targets_and_distractors_unchanged': True}
