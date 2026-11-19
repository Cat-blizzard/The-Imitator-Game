"""Repository-local runtime paths; no dependency on the original research host."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
REGISTRY = ROOT / 'runtime' / 'campaign_registry'


def configure():
    for path in (ROOT, ROOT / 'PolicyGen'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault('MS_ASSET_DIR', str(ROOT / 'assets' / 'maniskill'))
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    return ROOT


def local_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def evidence_path(value):
    """Resolve explicitly migrated, byte-identical evidence without editing it."""
    import json
    p = Path(value)
    aliases = ROOT / 'provenance/path_aliases.json'
    if aliases.exists():
        for old, relative in json.loads(aliases.read_text())['prefixes'].items():
            if p.is_relative_to(old):
                mapped = (ROOT / relative / p.relative_to(old)).resolve()
                if not mapped.is_relative_to(ROOT):
                    raise ValueError('Evidence mapping escaped repository')
                return mapped
    return p


def prepare_asset_catalog():
    """Create a relocation derivative; preserve source catalog and file hashes."""
    import json
    source = ROOT / 'assets/household/catalog.json'
    def visit(value):
        if isinstance(value, dict):
            return {str(evidence_path(k)) if k.startswith('/') else k: visit(v)
                    for k, v in value.items()}
        if isinstance(value, list): return [visit(v) for v in value]
        if isinstance(value, str) and value.startswith('/'): return str(evidence_path(value))
        return value
    value = visit(json.loads(source.read_text()))
    target = ROOT / 'runtime/household_catalog.json'
    target.parent.mkdir(exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if not target.exists() or target.read_text() != content: target.write_text(content)
    return target


configure()
