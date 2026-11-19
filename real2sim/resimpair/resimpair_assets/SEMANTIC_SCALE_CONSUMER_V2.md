# Source-aware semantic/scale consumer

The stable candidate is
`runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json`.
Pass it explicitly to sampler4 or the versioned production entry. Its
`containers` binding is exactly
`runs/container_affordance_consumer_20261030_v2/sampler_catalog.json`.
The original watcher, inventories, GPU queue, source IDs, partitions and cache
files remain unchanged.

The motivating source is metadata-named `Laptop`,
`objaverse:644c300e2244452c86d6fac0296477e1`. Its current native representation is
approximately 55 × 65 × 50 mm and 10.27 g. The old `electronics_controls` category
selected `small_electronics`; the old normalized OBB longest side, about 80.6 mm,
fell within that broad engineering range and was inherited. Cached grasp
qualification does not establish that the size corresponds to a human laptop.

The uniform sidecar scans all 58,117 raw metadata records and all 5,524 historical
native sources, then audits every final admitted cache entry. Whole-device
families, accessories, packaging and explicit miniature/toy contexts are separate
review cases. A whole-device size without matching source metric evidence is held
for review; no new representative dimensions or mass are guessed. These are
reviewable candidates, not name-derived semantic truth. No source-ID whitelist
or physical-success-based selection is used.

The frozen intermediate audit covered 856 admitted sources/862 variants. The
append-only final audit covers all 946 sources/952 variants in snapshot
`05524_01567_1793371238953079903`. The final consumer retains 935 sources,
941 variants and 26,634 cached poses; 11 source/variant entries are held with
their original bytes still available. It also removes 34 corresponding small
static-distractor descriptors and uses the independent strict container subset
of 159 variants/148 sources. Background and original physical cohorts are not
rewritten. Qualification reports are immutable standard bindings; their positive
flags refer only to consumer filtering/provenance, not semantic or physical success.

Across the full metadata population, 712 sources are review holds, of which
69 have historical native assets. These are not 712 confirmed errors. Many
sources have not reached native/grasp qualification. The audit found 1,524
historical engineering-size inheritances; this is also not an error count.
Seventeen old PolyHaven descriptors disagree with raw metadata partitions. Both
historical values are retained explicitly; the final admitted entries preserve
the current source partition. The first audit failure and its source code remain
available, followed by a v2 auditor that records these discrepancies instead of
stopping the census.

Read `audit_v2/summary.json`, `targeted_review_queue.csv`, and
`final_consumer_v2/final_admitted_audit.json` under the run directory. Reports and
real cached-entry regression evidence are in
`reports/semantic_scale_consumer_20261030_v1/`. The geometry preview was actually
inspected; it is an untextured CPU projection, not real-world dimensional proof.

Limits remain explicit. `Bench Hammer`, `Airplane Pencil Topper`, and similar
contexts can be potential false exclusions. Conversely, a whole-object title
containing an accessory word can bypass a lexical rule. A TV-stand mesh can be a
mounting bracket, and a little TV can be a toy. Held sources stay unknown until
source dimensions, entity/part structure and visual evidence resolve them.
The next revision should improve those evidence sources, not assign every laptop
a convenient hand-sized length. Corrections to actual geometry would need new
native/grasp qualification under a separately budgeted version.

Implementation: `semantic_scale_consumer_v1.py` supplies frozen classification
and filtering; `semantic_scale_consumer_v2.py` is the complete discrepancy-aware
auditor; `semantic_scale_publish_v2.py` validates the independent container review
and publishes the combined immutable bundle. All work here is CPU-only, without
new physical execution, cuDex jobs or paid API calls.
