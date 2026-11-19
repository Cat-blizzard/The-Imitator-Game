# Task asset selection candidate — 2026-10-21

This is an opt-in candidate in the SDG working copy. The original production
catalogs, original meshes, planner and acceptance thresholds are preserved.

The previous library is a discovery index, not a manipulation capability database.
`task_asset_selection.py` now separates three uses:

| Use | Required evidence | Output |
|---|---|---|
| Active task object | Exact category, supported mechanism, a passed independent historical verifier bound to the same asset files and actual episode role | Candidate for another small physical qualification; never automatically admitted |
| Static untouched distractor | Baked metric geometry, matching visual/collision bounds, finite geometry, conservative collision, source/license bindings | Existing `added_distractors` loader specification |
| Visual background | Hash-bound decoded diffuse/HDRI files, source and license | Existing candidate scene `background.wall/table/hdri` specification |

`physics_qualified` and `training_admitted` are **not** turned on by selection.
The active mechanism registry intentionally contains only `apple_bowl_transfer`.
It imports apple and bowl identities from a real passed episode, checks its actual
goal/program/embodiment and asset hashes, and forbids relabelling those records as
evidence for other objects or mechanisms. Apple grasping is not bowl grasping.
The selected asset's historical success does not imply that a new layout works.

## Request and output

```json
{
  "mechanism": "apple_bowl_transfer",
  "seed": 2026102100,
  "roles": {
    "selected_object": {
      "category": "apple", "max_extent_m": 0.06,
      "required_capabilities": ["grasp_lift_release"]
    },
    "source_container": {
      "category": "bowl", "required_capabilities": ["support_inside_cavity"]
    },
    "destination_container": {
      "category": "bowl", "required_capabilities": ["support_inside_cavity"]
    }
  },
  "distractors": {
    "count": 4,
    "permit_preflight_experiment": true,
    "allowed_categories": ["calculator", "hammer", "scissors", "flashlight"]
  }
}
```

The two container roles may use the same model, but receive different instance
IDs. A role with missing/unknown capabilities blocks the selection. An unsupported
count, insufficient distinct categories, modified file, mismatched sidecar or
ambiguous target-category distractor fails; the count is never silently reduced.
`max_extent_m` requires known metric geometry. Unknown dimensions do not pass.

```bash
PYTHONDONTWRITEBYTECODE=1 PolicyGen/.venv/bin/python \
  -m data_generation_methods.asset_library.task_asset_selection \
  --catalog /mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/static_catalog_v2/catalog.json \
  --request /mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/request_0.json \
  --historical-apple /mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e/layout2001_v2 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/selection_new.json
```

`proposal_distractor(asset, instance_id=..., center_xy=..., yaw_rad=...)` emits the
existing loader fields, including the **world bounding-box center**, static body
type, baked unit scale and hash-bound files. Placement still needs the existing
scene screen, native-object collision checks and strict planner. This module does
not claim that arbitrary clutter placement is feasible.

`discover_raw(index, terms=[...], sources=[...])` searches local names/categories.
Its results explicitly remain `raw_discovery_only`. A matching name never becomes
an executable adapter without preparation and mechanism-specific qualification.

## New preparation and reusable materials

`prepare_selection_candidate.py` prepared 10 new categories from local Objaverse
and ABO GLBs with the established household geometry code. The first wrapper
attempt correctly failed when it tried to overwrite the immutable `asset.json`.
That attempted run is retained. The corrected run writes a separate
`qualification_asset.json`; all 10 then passed CPU geometry preflight.

`prepare_scanned_static_candidate.py` also prepared a GSO tape roll from the
original OBJ/MTL/texture package. It resolves only unambiguous material filenames
inside that original package, binds all originals, and exports a self-contained
GLB without changing originals. The hole is filled by conservative static
collision, so this asset cannot be used for insertion/threading/containment.

11 new distinct categories plus the old 12 gives 23 **static CPU-preflight
categories**, not 23 active manipulation assets. CPU previews excluded headphones
and the very thin paintbrush from the first simulation groups because their
visible semantics were weak. The GSO tape is an additional prepared candidate;
it did not enter those two already fixed physical groups. A later CPU review
noticed that the thin wall clock, placed upright on the table, relies on static
anchoring rather than a natural stand. The physical run already used it, so that
run is retained with this realism limitation. A future-only selection replaces
it with the GSO tape; this is not a retrospective improvement to that run.

`background_candidates.py` binds/decodes all 29 existing diffuse surfaces and
provides three wall/table/HDRI combinations. The old 15 HDRIs were mostly abandoned
or industrial locations. Three additional official CC0 normal indoor panoramas
were downloaded at 1K, sequentially, with publisher MD5 checks and SHA256 records:

- [Small Empty Room 3](https://polyhaven.com/a/small_empty_room_3)
- [Small Empty Room 4](https://polyhaven.com/a/small_empty_room_4)
- [Poly Haven Studio](https://polyhaven.com/a/poly_haven_studio)

Downloaded HDR payload: 4,789,274 bytes; declared cumulative limit 200 MiB; API
charges zero. The normal indoor visual-background pool is expanded, not the
physical environment collision pool. Surface appearance does not alter friction
or other contact parameters. CPU galleries are not simulator render evidence.

## Evidence and validation

Artifacts are under
`/mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/`:

- `combined_catalog.json`: 11 adapters with provenance and scoped qualification.
- `selection_manifest.json`: fixed development/held-out category groups, four
  distractors each; placement examples are not a physics verdict.
- `next_round_selection_manifest_v3.json`: future-only clock-to-tape replacement.
  `selection_v2_timing_correction.json` explicitly corrects the intermediate v2
  draft's mistaken claim that the replacement preceded the already completed
  holdout execution. No historical execution or original manifest was modified.
- `backgrounds/background_cases.json`: runtime input interface and full download
  receipts. `backgrounds/gallery.jpg` previews all three appearance choices.
- `static_catalog_v2/gallery.jpg`, `gso_static/tape_roll/preview.jpg`: CPU geometry.
- `raw_role_discovery.json`: candidate retrieval examples with no automatic admission.
- `geometry_summary.json`: counts and scope; actual scene outcomes are maintained
  by the separate simulation runner, not retroactively written into source assets.

Run the failure-oriented tests using `PYTHONDONTWRITEBYTECODE=1`, a writable SDG
`TMPDIR`, and `pytest -p no:cacheprovider`. They include real local asset geometry
reload and malicious/mistaken relabelling, hash change, duplicate geometry,
insufficient diversity, unknown mechanism and size-limit counterexamples.
