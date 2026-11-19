# Static-state acceptance v1.0

This independent gate complements unchanged `single_actor_v2.1`. An accepted
person structure report does not establish task, object, or state preservation.
The gate validates explicit observations and immutable media/context bindings;
it does not infer visual correctness or measure detector accuracy.

`static_state_acceptance.gate_state(report_path, image_path, package_path,
expected_context)` requires a context derived from the candidate's verified
receipt → proposal → human specification/window → condition packet chain.
The review cannot supply its own expected context. The full schema is
`schemas/static_state_review_v1.schema.json`; unknown or omitted fields fail.

The binding includes exact human source/session/video hash, interval/fps/split,
execution ID, camera ID, condition packet hash, reference image hash, condition
role, and stable selected-object/source/destination entity IDs. A matching actor
plan alone cannot substitute for this package binding.

Seven dimensions remain separate: static task correspondence, object identity,
relative scale, role mapping, visible state relations, non-target preservation,
and camera correspondence. Every pass/fail needs identified direct visual or
human evidence bound to the actual decoded image. An object having the same
color is insufficient identity evidence. Ambiguous or hidden correspondence
remains unknown. VLM-only affirmative answers remain unknown.

The static state verdict does not certify action order, continuity, contact
forces, human 3D labels, actor identity, or full-video quality. It returns
`clears_static_state`; its `can_generate_video` always remains false. The unified
candidate separately combines this with the actor gate, request/result
provenance, and an explicit cumulative budget. No offline test fixture or copied
simulation reference is a newly accepted synthetic-human image.

The first review uses role `initial`. A terminal review must independently bind
its terminal context. The initial generation request cannot consume a terminal
image or future-state/event instructions. Single-operator placement can change
hand entry while keeping task roles and visible initial relations.

Protocol and actual-source negative controls are under
`/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/independent_review/`.
The g00/g01 human sources are different IDs and sessions, correcting a v1
reporting mistake. Both are historical reviewed sources; this local probe is
not a new blind population validation. Old evidence is retained unchanged.
