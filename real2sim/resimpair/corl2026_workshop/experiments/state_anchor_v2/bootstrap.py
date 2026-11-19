"""Verify repository-local dependency bytes; retain original provenance unchanged."""
import hashlib
import json
from pathlib import Path
import sys
from resimpair_runtime import ROOT, configure


def activate():
    configure()
    manifest_path = ROOT / 'provenance/original_data_generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    canonical = ROOT / 'data_generation_methods'
    # Imitator 2.0 port: only the Python sources of the frozen tree are vendored (its images, logs and run results
    # are not). Every vendored file must still match the original bytes; missing non-code files are counted.
    verified, missing = 0, []
    for row in manifest['files']:
        relative = Path(row['relative']).relative_to('code/data_generation_methods')
        if '..' in relative.parts:
            raise ValueError('Invalid dependency manifest path')
        path = canonical / relative
        if not path.exists():
            if relative.suffix == '.py':
                raise ValueError('Vendored dependency missing: ' + str(relative))
            missing.append(str(relative))
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != row['sha256']:
            raise ValueError('Vendored dependency changed: ' + str(relative))
        verified += 1
    if not manifest['files']:
        raise ValueError('Empty dependency manifest')
    module = sys.modules.get('data_generation_methods')
    if module is not None:
        origins = [Path(p).resolve() for p in getattr(module, '__path__', ())]
        if not origins or origins[0] != canonical:
            raise RuntimeError('Another data_generation_methods tree was imported')
    sys.path.insert(0, str(ROOT))
    return {'source_root': str(canonical), 'manifest': str(manifest_path),
            'verified_file_count': verified, 'not_vendored_non_code_files': len(missing), 'repository_local': True,
            'vendored_bytes_match_original': True,
            'original_source_root_for_provenance_only': manifest['source_root']}
