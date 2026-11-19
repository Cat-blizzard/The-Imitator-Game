"""Static success-predicate lint for ManiSkill-style tasks (evaluate() returning a dict with 'success').

Promoted from research/scripts/predicate_lint.py. Known limitations: handles evaluate() only (RoboTwin-style
check_success() not yet), names-based boolean heuristics, single-file analysis (no cross-method dataflow).

For every task file, inside evaluate():
  - builds the dependency closure of the `success` expression(s) over local names and self.<attr> latches
  - UNUSED_COND: boolean-looking locals (is_*/has_*/*_ok/*enough/...) that are computed but never reach success
  - EVER_LATCH: success depends on a self.<x> updated as `self.x = self.x | cond` (once true, stays true)
  - NO_STATIC_CHECK: success does not depend on any static/settled check (is_*static*, *settled*, ~is_grasped)
usage: python -I -B predicate_lint.py TASK_DIR [--json out.json]
"""
import ast, json, re, sys, glob, os

BOOLISH = re.compile(r"^(is_|has_|.*_ok$|.*enough$|.*_placed$|.*_tilted$|.*inside|.*_static$|currently_|.*_reached)")

def names_in(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            out.add("self." + n.attr)
    return out

def target_key(t):
    if isinstance(t, ast.Name): return t.id
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self": return "self." + t.attr
    return None

def lint(path):
    src = open(path).read(); tree = ast.parse(src)
    ev = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "evaluate"]
    if not ev: return None
    ev = ev[0]
    deps, assigned_at = {}, {}
    succ_exprs = []
    for st in ast.walk(ev):
        if isinstance(st, (ast.Assign, ast.AugAssign)):
            tgts = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in tgts:
                k = target_key(t)
                if k:
                    deps.setdefault(k, set()).update(names_in(st.value))
                    assigned_at[k] = st.lineno
                    if k == "success": succ_exprs.append(st.value)
        if isinstance(st, ast.keyword) and st.arg == "success":
            succ_exprs.append(st.value); deps.setdefault("success", set()).update(names_in(st.value))
        if isinstance(st, ast.Dict):  # return {"success": ...} style (upstream ManiSkill)
            for kk, vv in zip(st.keys, st.values):
                if isinstance(kk, ast.Constant) and kk.value == "success":
                    succ_exprs.append(vv); deps.setdefault("success", set()).update(names_in(vv))
    closure, stack = set(), ["success"]
    while stack:
        k = stack.pop()
        for d in deps.get(k, ()):
            if d not in closure:
                closure.add(d); stack.append(d)
    boolish = sorted(k for k in deps if not k.startswith("self.") and BOOLISH.match(k) and k != "success")
    unused_all = [k for k in boolish if k not in closure]
    generic = re.compile(r"(robot_static|left_static|right_static|^is_grasped$|_grasped$|^is_grasped_(left|right)$|^is_[a-z0-9_]*_grasped$)")
    unused = [k for k in unused_all if not generic.search(k)]
    seg = {}
    for st in ast.walk(ev):
        if isinstance(st, (ast.Assign, ast.AugAssign)):
            for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                k = target_key(t)
                if k: seg.setdefault(k, []).append(ast.get_source_segment(src, st.value) or "")
    latches = sorted(k for k in closure if k.startswith("self.") and k in deps and k in deps[k])
    texts = [x for k in closure | {"success"} for x in seg.get(k, [])] + [ast.get_source_segment(src, e) or "" for e in succ_exprs]
    static_ok = any(re.search(r"static|settle|released", k) for k in closure) or \
        any(re.search(r"(~\s*\(?\s*is_\w*grasp|logical_not\(\s*is_\w*grasp|not_grasped|is_static|\.is_static\()", t) for t in texts)
    return dict(task=os.path.basename(path)[:-3], unused_conditions=unused, unused_generic=[k for k in unused_all if k not in unused],
                ever_latches=latches, no_settled_check=not static_ok, n_conditions=len(boolish))

def flags(row: dict | None) -> list[str]:
    """Flag names for one lint row (keys of audit.primitives.LINT_FLAG_TO_WEAKNESS)."""
    if not row:
        return []
    out = []
    if row["unused_conditions"]: out.append("UNUSED_COND")
    if row["ever_latches"]: out.append("EVER_LATCH")
    if row["no_settled_check"]: out.append("NO_STATIC_CHECK")
    return out


if __name__ == "__main__":
    d = sys.argv[1]; rows = [r for r in (lint(f) for f in sorted(glob.glob(f"{d}/_0*.py"))) if r]
    if "--json" in sys.argv: json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    for r in rows:
        flags = []
        if r["unused_conditions"]: flags.append("SEMANTIC_UNUSED:" + ",".join(r["unused_conditions"]))
        if r["ever_latches"]: flags.append("EVER_LATCH:" + ",".join(x[5:] for x in r["ever_latches"]))
        if r["no_settled_check"]: flags.append("NO_SETTLED_CHECK")
        print(f"{r['task']:38s} {' | '.join(flags)}")
    n = len(rows)
    print(f"\n{n} tasks | SEMANTIC_UNUSED {sum(bool(r['unused_conditions']) for r in rows)} | EVER_LATCH {sum(bool(r['ever_latches']) for r in rows)} | NO_SETTLED_CHECK {sum(r['no_settled_check'] for r in rows)}")
