"""Goals are separate from layouts. External links are family-level, not positives."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ObjectSpec:
    category: str
    model_id: int
    name: str
    largest_extent_m: float
    rotation_x_deg: float = 90.


@dataclass(frozen=True)
class SceneSpec:
    name: str
    objects: tuple
    agibot_tasks: tuple
    egodex_families: tuple
    description: str


SCENES = {
    'grocery_fruit': SceneSpec('grocery_fruit', (
        ObjectSpec('035_apple', 1, 'apple', .055),
        ObjectSpec('103_fruit', 1, 'pear', .065)),
        (327, 354, 356, 385), ('basic_pick_place', 'declutter_desk'),
        'Pick fruit, pack fruit into a container, or sort fruit into two containers.'),
    'grocery_drinks': SceneSpec('grocery_drinks', (
        ObjectSpec('071_can', 0, 'can', .080),
        ObjectSpec('038_milk-box', 0, 'milk_carton', .095)),
        (327, 354, 356, 360, 372), ('basic_pick_place', 'declutter_desk'),
        'Pick, pack, or sort drink containers.'),
    'electronics': SceneSpec('electronics', (
        ObjectSpec('061_battery', 0, 'battery', .075),
        ObjectSpec('079_remotecontrol', 0, 'remote', .105)),
        (376, 377, 380), ('basic_pick_place', 'declutter_desk'),
        'Pick, pack, or sort small electronics.'),
    'personal_care': SceneSpec('personal_care', (
        ObjectSpec('049_shampoo', 1, 'shampoo', .095),
        ObjectSpec('080_pillbottle', 1, 'pill_bottle', .075)),
        (365, 374, 377), ('basic_pick_place', 'declutter_desk'),
        'Pick, pack, or sort personal-care products.'),
    'tableware': SceneSpec('tableware', (
        ObjectSpec('039_mug', 1, 'mug', .085),
        ObjectSpec('021_cup', 0, 'cup', .070)),
        (357, 378), ('basic_pick_place', 'clean_cups', 'clean_tableware'),
        'Move and collect tableware; this does not label the washing step.'),
    'desk_books': SceneSpec('desk_books', (
        ObjectSpec('043_book', 0, 'book', .100),
        ObjectSpec('081_playingcards', 0, 'cards', .085)),
        (384, 378), ('basic_pick_place', 'declutter_desk'),
        'Collect desk objects into trays; shelving is a separate future goal.'),
}

# Same scene and seed for these programs. No program identifier enters observations.
# (object index, destination container index), executed in this explicit order.
PROGRAMS = {
    'pick_first_to_left': ((0, 0),),
    'pick_first_to_right': ((0, 1),),
    'pick_second_to_left': ((1, 0),),
    'pick_second_to_right': ((1, 1),),
    'sort_direct': ((0, 0), (1, 1)),
    'sort_cross': ((0, 1), (1, 0)),
    'pack_left': ((0, 0), (1, 0)),
    'pack_right': ((0, 1), (1, 1)),
}


def goal_key(scene, program):
    return f'expansion:{scene}:{program}'


def inventory():
    return dict(scenes={k: asdict(v) for k, v in SCENES.items()}, programs=PROGRAMS,
                exact_goals=len(SCENES) * len(PROGRAMS),
                external_pair_policy='Family links are retrieval candidates, not certified goal-equivalent positives.',
                original_benchmark_environment_ids_unchanged=True)
