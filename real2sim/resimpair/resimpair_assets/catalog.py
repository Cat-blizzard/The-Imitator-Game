"""Metadata retrieval is a candidate generator, never an affordance certificate."""
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from resimpair_grasp.common import ROOT, binding, read, sha, write


def categories(value):
    if value is None: return []
    if isinstance(value, str): return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value): return value
    return []


def classify(row, policy):
    cats = categories(row.get("categories"))
    name = re.sub(r"[_\-/]+", " ", row.get("name", "")).casefold()
    if row.get("kind") != "model": return None, "not_a_mesh_model"
    if set(cats) & set(policy["exclude_categories"]): return None, "large_or_unsupported_category"
    if any(re.search(p, name) for p in policy["exclude_name_patterns"]): return None, "ambiguous_or_unsupported_name"
    hits = []
    for i, spec in enumerate(policy["categories"]):
        matched = [p for p in spec["patterns"] if re.search(p, name)]
        source = sorted(set(cats) & set(spec["source_categories"]))
        vocabulary=policy.get('literal_name_tokens',{}).get(spec['id'])
        if vocabulary is not None and set(re.findall('[a-z]+',name))-set(vocabulary):
            matched=[]
        if matched or source: hits.append((spec.get('name_priority',100-i) if matched else 1, -i, spec, matched, source))
    if not hits: return None, "no_supported_tabletop_role"
    _, _, spec, matched, source = max(hits, key=lambda x: (x[0], x[1]))
    matched_categories=sorted({h[2]['id'] for h in hits if h[3]})
    ambiguities=[p for p in policy.get('manipulation_ambiguous_patterns',[]) if re.search(p,name)]
    if len(matched_categories)>1:ambiguities.append('multiple_named_categories:'+','.join(matched_categories))
    return {"category": spec["id"], "longest_m": spec["longest_m"],
            "manipulation_candidate": spec["manipulation_candidate"] and not ambiguities,
            "semantic_ambiguities":ambiguities,
            "semantic_status": policy["semantic_status"],
            "evidence": {"name_patterns": matched, "source_categories": source}}, None


def build(source, policy_path, output):
    if output.exists(): raise ValueError("Index exists; use a new version")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = read(policy_path); totals, accepted, reasons = Counter(), Counter(), Counter()
    with sqlite3.connect(output) as db:
        db.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, source TEXT, name TEXT, category TEXT, role_candidate INTEGER, partition TEXT, source_sha TEXT, path TEXT, bytes INTEGER, eligible INTEGER, reason TEXT, record TEXT, classification TEXT)")
        db.execute("CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, name, categories)")
        with source.open() as f:
            for line in f:
                row = json.loads(line); totals[row["source"]] += 1
                classification, reason = classify(row, policy)
                path = Path(row["primary_path"]) if row.get("primary_path") else None
                exists = path is not None and path.is_file()
                if not reason and not exists: reason = "primary_mesh_missing"
                if not reason and path.suffix.lower() not in {".glb", ".gltf", ".obj", ".ply", ".stl"}: reason = "unsupported_mesh_format"
                size = path.stat().st_size if exists else 0
                if not reason and size > 24 * 2**20: reason = "input_exceeds_24MiB_budget"
                source_sha = row.get("sha256") or ""
                group = source_sha or row["asset_id"]
                partition = "heldout_asset" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "development"
                # Keep preparation/provenance fields; the original index hash
                # binds verbose prior validation without duplicating it here.
                compact = {k:row.get(k) for k in ['asset_id','source','source_id','name','categories','license',
                           'primary_path','portable_primary_path','sha256','kind','source_record_path','url','author'] if k in row}
                db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    row["asset_id"], row["source"], row.get("name", ""), (classification or {}).get("category"),
                    int((classification or {}).get("manipulation_candidate", False)), partition, source_sha,
                    str(path) if path else None, size, int(reason is None), reason,
                    json.dumps(compact), json.dumps(classification)))
                db.execute("INSERT INTO search VALUES (?,?,?)", (row["asset_id"], row.get("name", ""), " ".join(categories(row.get("categories")))))
                if reason: reasons[reason] += 1
                else: accepted[classification["category"]] += 1
        db.execute("CREATE INDEX category_eligibility ON assets(eligible,category,partition,role_candidate)")
    result = {"schema": "raw_role_catalog_v1", "source": binding(source), "policy": binding(policy_path),
              "source_counts": totals, "metadata_candidate_counts": accepted, "rejected_metadata_counts": reasons,
              "builder":binding(__file__),"qualified_for_simulation": 0, "scope": "metadata and file checks only; no geometry, visual semantics or manipulation inferred"}
    write(output.with_suffix(".json"), result)
    return result


def search(index, *, category=(), source=(), role="static", partition=None, query=None, limit=20):
    if role not in {"static", "manipulation"}: raise ValueError("Unknown role")
    filters, params = ["eligible=1"], []
    for name, values in [("category", category), ("source", source)]:
        if values: filters.append(name + " IN (" + ",".join("?" for _ in values) + ")"); params.extend(values)
    if partition: filters.append("partition=?"); params.append(partition)
    if role == "manipulation": filters.append("role_candidate=1")
    if query:
        filters.append("id IN (SELECT id FROM search WHERE search MATCH ?)")
        params.append('"' + query.replace('"', '""') + '"')
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        sql = "SELECT * FROM assets WHERE " + " AND ".join(filters) + " ORDER BY id"
        if limit is not None: sql += " LIMIT ?"; params.append(limit)
        rows = [dict(r) for r in db.execute(sql, params)]
    for r in rows:
        r["record"] = json.loads(r["record"]); r["classification"] = json.loads(r["classification"])
    return rows


def balanced(rows, count, seed, exclude=()):
    """Balance categories then sources; deduplicate exact known source bytes."""
    groups = defaultdict(list); excluded = set(exclude)
    for r in rows:
        if r["id"] not in excluded:
            groups[r["category"], r["source"]].append(r)
    queues = {}
    for k, values in groups.items():
        values.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        queues[k] = deque(values)
    order = sorted(queues); result, seen = [], set()
    # Interleave one source for each category in each round, rather than
    # overweighting a category just because it exists in more source datasets.
    per_category = defaultdict(deque)
    for k in order: per_category[k[0]].append(k)
    while len(result) < count and any(queues.values()):
        for category in sorted(per_category):
            keys = per_category[category]
            for _ in range(len(keys)):
                key = keys[0]; keys.rotate(-1)
                if queues[key]: break
            else: continue
            row = queues[key].popleft(); identity = row["source_sha"] or row["id"]
            if identity in seen: continue
            seen.add(identity); result.append(row)
            if len(result) == count: break
    return result
