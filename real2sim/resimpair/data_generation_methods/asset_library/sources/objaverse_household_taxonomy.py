"""Exact LVIS household labels plus conservative name/tag retrieval aliases.

These are retrieval categories, not rendered semantic or physical approval.
"""
import re

GROUPS = {
    "drinkware": "Dixie_cup cup mug teacup measuring_cup glass_(drink_container) flute_glass shot_glass wineglass chalice canteen thermos_bottle",
    "bowls_plates": "bowl soup_bowl sugar_bowl plate salad_plate paper_plate dish saucer platter chinaware",
    "bottles_jars": "bottle beer_bottle water_bottle wine_bottle jar canister milk_can can aerosol_can perfume atomizer shaker saltshaker pepper_mill",
    "containers_storage": "box basket bucket carton crate barrel bread-bin cigar_box pencil_box hatbox toolbox clothes_hamper hamper trash_can cooler_(for_food) wine_bucket urn tank_(storage_vessel) canister packet pouch trunk",
    "books_stationery": "Bible book hardback_book paperback_book booklet notebook notepad diary magazine comic_book phonebook newspaper binder clipboard envelope pencil pen marker Sharpie eraser ruler measuring_stick paperweight pencil_sharpener stapler_(stapling_machine) puncher tape_(sticky_cloth_or_paper) duct_tape rubber_band clip thumbtack",
    "cutlery_utensils": "fork knife steak_knife spoon soupspoon wooden_spoon ladle spatula cooking_utensil chopstick tongs skewer stirrer straw_(for_drinking) peeler_(tool_for_fruit_and_vegetables) grater eggbeater whisk strainer colander funnel can_opener corkscrew bottle_opener nutcracker reamer_(juicer)",
    "cookware": "pan_(for_cooking) pan_(metal_container) frying_pan saucepan pot wok casserole kettle teakettle teapot coffeepot pitcher_(vessel_for_liquid) cream_pitcher gravy_boat tray chopping_board rolling_pin grill hotplate potholder",
    "lids_handles_hardware": "bottle_cap cork_(bottle_plug) cover handle knob doorknob hinge latch deadbolt hook padlock combination_lock key bolt nut clothespin",
    "cleaning": "sponge scrubbing_brush broom mop dustpan vacuum_cleaner soap cleansing_agent detergent dishwasher_detergent freshener dispenser toilet_tissue paper_towel dishrag dishtowel brush scraper",
    "bath_personal": "toothbrush toothpaste hairbrush comb hair_dryer shaver_(electric) shaving_cream shampoo lip_balm nailfile hand_glass shower_head shower_curtain shower_cap first-aid_kit Band_Aid tissue_paper",
    "textiles": "bath_towel hand_towel towel bath_mat doormat tablecloth napkin place_mat apron cushion pillow blanket quilt bedspread bedspread curtain runner_(carpet) mattress bathrobe handkerchief potholder",
    "furniture_seating": "chair armchair folding_chair rocking_chair deck_chair highchair chaise_longue recliner sofa sofa_bed loveseat bench stool footstool step_stool ottoman beanbag futon music_stool",
    "furniture_storage": "cabinet cupboard drawer dresser wardrobe armoire bookcase file_cabinet locker coatrack spice_rack towel_rack coat_hanger headboard",
    "furniture_tables_beds": "table coffee_table dining_table kitchen_table desk bed bunk_bed crib ironing_board easel bulletin_board pegboard blackboard corkboard",
    "appliances": "refrigerator oven microwave_oven toaster toaster_oven coffee_maker blender mixer_(kitchen_tool) food_processor cooker stove dishwasher automatic_washer iron_(for_clothing) sewing_machine waffle_iron ice_maker water_heater air_conditioner heater fan radiator water_cooler fume_hood",
    "electronics_controls": "remote_control telephone cellular_telephone laptop_computer computer_keyboard mouse_(computer_equipment) mousepad monitor_(computer_equipment) television_set radio_receiver stereo_(sound_system) speaker_(stero_equipment) subwoofer CD_player record_player printer router_(computer_equipment) calculator camera camera_lens webcam joystick earphone headset battery thermostat timer alarm_clock wall_clock clock watch pocket_watch thermometer scale_(measuring_instrument)",
    "lighting_decor": "lamp table_lamp oil_lamp lampshade chandelier lantern candle candle_holder flashlight lightbulb hourglass vase flowerpot pottery flower_arrangement bouquet picture painting mirror wind_chime chime bell globe piggy_bank photo_frame window_box_(for_plants) aquarium fishbowl birdcage birdfeeder watering_can",
    "home_fixtures": "kitchen_sink sink washbasin faucet water_faucet bathtub toilet urinal cistern fireplace wall_socket vent mail_slot mailbox_(at_home) door_window",
    "hand_tools": "hammer mallet screwdriver wrench pliers scissors shears drill handsaw file_(tool) paintbrush scraper tape_measure ladder stepladder sawhorse crowbar clippers_(for_plants) shovel garden_hose hose",
    "bags_cases": "grocery_bag shopping_bag plastic_bag tote_bag handbag shoulder_bag backpack satchel briefcase suitcase duffel_bag wallet cigarette_case gift_wrap",
    "food_props": "apple banana orange_(fruit) mandarin_orange pear lemon lime kiwi_fruit peach apricot plum strawberry raspberry blackberry blueberry grape pineapple watermelon melon cantaloup avocado coconut pomegranate papaya persimmon date_(fruit) fig_(fruit) cherry tomato cucumber eggplant carrot potato sweet_potato onion garlic ginger zucchini bell_pepper chili_(vegetable) broccoli cauliflower celery lettuce cabbage pumpkin radish turnip mushroom egg boiled_egg bread baguet bun bagel biscuit cookie cracker toast_(food) butter cheese honey jam condiment salt pepper olive_oil vinegar Tabasco_sauce hot_sauce",
}
GROUPS = {group: tuple(dict.fromkeys(words.split())) for group, words in GROUPS.items()}

# Name/tag aliases catch useful objects omitted from the 46k LVIS subset.
NAME_TERMS = {
    "drinkware": "mug|teacup|coffee cup|drinking cup|drinking glass|wine glass|goblet|tumbler|measuring cup",
    "bowls_plates": "bowl|dinner plate|dinnerplate|salad plate|saucer|platter|tableware|ceramic plate",
    "bottles_jars": "bottle|canister|mason jar|glass jar|storage jar|salt shaker|pepper mill|thermos",
    "containers_storage": "storage box|cardboard box|cardboardbox|wooden box|plastic box|basket|bucket|bread bin|trash can|trashcan|garbage bin|storage container|toolbox|wooden crate|plastic crate",
    "books_stationery": "book|notebook|hardcover|hardback|paperback|clipboard|stapler|pencil|ballpoint|eraser|stationery|scotch tape|paper clip|paperclip|envelope",
    "cutlery_utensils": "spoon|fork|cutlery|ladle|spatula|chopstick|chopsticks|kitchen knife|butter knife|chef knife|kitchen utensil|kitchen utensils|can opener|bottle opener|corkscrew|colander|grater|tongs|whisk|strainer|vegetable peeler",
    "cookware": "frying pan|fryingpan|saucepan|skillet|cooking pot|stockpot|cookware|teapot|teakettle|kettle|coffeepot|pitcher|cutting board|chopping board|rolling pin|baking tray|serving tray|baking dish|wok",
    "lids_handles_hardware": "bottle cap|bottlecap|jar lid|pot lid|cup lid|door handle|drawer handle|doorknob|door knob|cabinet handle|cabinet hinge|padlock|combination lock|clothespin",
    "cleaning": "sponge|scrubbing brush|scrub brush|cleaning brush|broom|dustpan|vacuum cleaner|soap|dish detergent|dishwashing liquid|spray cleaner|cleaning mop|floor mop|toilet brush",
    "bath_personal": "toothbrush|toothpaste|hairbrush|hair brush|hairdryer|hair dryer|shampoo|shaving cream|electric shaver|shower head|tissue box|tissue paper",
    "textiles": "towel|bath mat|bathmat|doormat|tablecloth|napkin|cushion|pillow|blanket|quilt|curtain|mattress|oven mitt|apron",
    "furniture_seating": "chair|armchair|recliner|sofa|couch|loveseat|ottoman|footstool|bar stool|barstool|folding stool|wooden bench|dining bench",
    "furniture_storage": "cabinet|cupboard|drawer|dresser|wardrobe|armoire|bookcase|bookshelf|shelving unit|storage shelf|storage shelves|spice rack|towel rack|coat rack|coat hanger|nightstand|bedside table",
    "furniture_tables_beds": "coffee table|dining table|kitchen table|side table|end table|wooden table|round table|office desk|writing desk|computer desk|bedframe|bed frame|bunk bed|single bed|double bed|crib|ironing board",
    "appliances": "refrigerator|fridge|microwave|toaster|coffee maker|coffeemaker|blender|stand mixer|food processor|dishwasher|washing machine|washer dryer|sewing machine|electric kettle|air conditioner|electric heater|table fan|ceiling fan|waffle iron|electric stove|gas stove|cooking oven|range hood",
    "electronics_controls": "remote control|remotecontroller|television remote|tv remote|cellphone|smartphone|telephone|laptop|computer keyboard|computer mouse|computer monitor|television|desktop printer|alarm clock|wall clock|kitchen timer|thermostat|calculator|wireless router|game controller|webcam|headphones|earphones|desktop speaker|bluetooth speaker",
    "lighting_decor": "table lamp|desk lamp|floor lamp|pendant lamp|lampshade|chandelier|lantern|candle holder|candleholder|lightbulb|light bulb|vase|flowerpot|flower pot|plant pot|watering can|wall mirror|hand mirror|picture frame|photo frame|piggy bank|hourglass|fishbowl|aquarium",
    "home_fixtures": "kitchen sink|bathroom sink|washbasin|wash basin|faucet|water tap|bathtub|bath tub|toilet|shower head|wall socket|electrical outlet|light switch|kitchen counter|countertop",
    "hand_tools": "hammer|mallet|screwdriver|wrench|pliers|scissors|handsaw|hand saw|paintbrush|paint brush|tape measure|measuring tape|cordless drill|power drill|step ladder|stepladder|garden shovel|garden hose|putty knife",
    "bags_cases": "shopping bag|grocery bag|plastic bag|paper bag|tote bag|handbag|backpack|briefcase|suitcase|duffel bag|wallet",
    "food_props": "apple|banana|orange fruit|mandarin orange|pear fruit|lemon|lime fruit|kiwi fruit|peach|apricot|strawberry|raspberry|blackberry|blueberry|pineapple|watermelon|cantaloupe|avocado|coconut|tomato|cucumber|eggplant|carrot|potato|onion|garlic|ginger|zucchini|bell pepper|broccoli|cauliflower|lettuce|pumpkin|radish|mushroom|boiled egg|bread loaf|baguette|bagel|burger bun|cookie|butter stick|cheese block|honey jar|jam jar|olive oil|vinegar bottle",
}
PATTERNS = {g: re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t).replace(r"\ ", r"[ _-]+")
                         for t in terms.split("|")) + r")(?![a-z0-9])", re.I)
            for g, terms in NAME_TERMS.items()}
NEGATIVE = re.compile(r"\b(spongebob|squarepants|minecraft|fortnite|roblox|spaceship|starship|battlefield|diorama|character|avatar|terrain|landscape|cityscape|skybox|weapon)\b", re.I)


def name_groups(annotation):
    name = str(annotation.get("name", ""))
    tags = " ".join(str(t.get("name", "")) if isinstance(t, dict) else str(t)
                    for t in annotation.get("tags", []))
    if NEGATIVE.search(name + " " + tags):
        return [], "explicit_unrelated_scene_character_or_game_title"
    named = [g for g, pattern in PATTERNS.items() if pattern.search(name)]
    # "blender" is a ubiquitous 3D-authoring software tag. It does not identify
    # a kitchen blender; keep a literal named/LVIS blender, never this tag alone.
    clean_tags = re.sub(r"(?<![a-z0-9])blender(?![a-z0-9])", "", tags, flags=re.I)
    tagged = [g for g, pattern in PATTERNS.items() if pattern.search(clean_tags)]
    return sorted(set(named + tagged)), "name_supported" if named else "tags_only" if tagged else "LVIS_only"
