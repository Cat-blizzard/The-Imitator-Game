"""Hash-bound, scalable source/role/split lookup for optional expanded catalogs.

The small request contains bindings, not thousands of inline exclusions. The
producer freezes this index before geometry/physics. Unknown sources and
partition conflicts are denied; descriptors retain their original partition.
"""
from functools import lru_cache
import numpy as np

from resimpair_grasp.common import read, resolve

SCHEMA = 'source_role_catalog_v1'


@lru_cache(maxsize=8)
def _load(catalog_path, catalog_hash, index_path, index_hash):
    catalog_ref = {'path': catalog_path, 'sha256': catalog_hash}
    index_ref = {'path': index_path, 'sha256': index_hash}
    catalog = read(resolve(catalog_ref)); index = read(resolve(index_ref))
    if index.get('schema') != SCHEMA or index.get('catalog') != catalog_ref:
        raise ValueError('source_role_index_catalog_mismatch')
    sources = index['sources']; rows = []; seen = set()
    for bound in catalog['assets']:
        descriptor = read(resolve(bound)); source = descriptor['source_id']
        policy = sources.get(source)
        if policy is None:
            raise ValueError('catalog_source_absent_from_role_index')
        if source in seen:
            raise ValueError('duplicate_catalog_source')
        seen.add(source)
        if bound not in policy.get('descriptors', []):
            raise ValueError('descriptor_absent_from_role_index')
        if not policy.get('allowed_as_distractor'):
            continue
        if policy.get('partitions') != [descriptor['partition']]:
            raise ValueError('catalog_source_partition_conflict')
        if not descriptor.get('roles', {}).get('static_distractor'):
            continue
        dimensions = np.asarray(descriptor['dimensions_m'], dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('invalid_distractor_dimensions')
        rows.append((bound, descriptor))
    return tuple(rows)


def select(catalog, index, partition, maximum, excluded):
    # Verify the two manifest bindings on every call, including cache hits.
    resolve(catalog); resolve(index)
    rows = _load(catalog['path'], catalog['sha256'], index['path'], index['sha256'])
    return [(b, d) for b, d in rows if d['source_id'] not in excluded
            and d['partition'] == partition and max(d['dimensions_m']) <= maximum]
