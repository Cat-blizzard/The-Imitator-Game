"""Select cached assets by partition, category and metric bounds, without physics."""
import argparse
import hashlib
import json
from pathlib import Path
from resimpair_grasp.common import binding, read, resolve, write


def select(catalog, output, count, partition, categories, maximum_dimension, seed):
    if output.exists():
        raise ValueError('Selection exists; preserve the previous selection')
    if not 1 <= count <= 10000 or not 0 < maximum_dimension <= .5:
        raise ValueError('Finite asset count and metric size required')
    choices = []
    for row in read(catalog)['rows']:
        if row['status'] != 'offline_tabletop_feasible' or row['partition'] != partition:
            continue
        if categories and row['category'] not in categories:
            continue
        asset = read(resolve(row['asset']))
        variant = next(v for v in asset['variants'] if v['id'] == asset['selected_variant'])
        if max(variant['dimensions_m']) <= maximum_dimension:
            choices.append(row)
    choices.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['source_id']}".encode()).hexdigest())
    if count > len(choices):
        raise ValueError(f'Requested {count} assets; only {len(choices)} match the query')
    output.mkdir(parents=True)
    write(output/'plan.json', {
        'schema': 'tabletop_query_plan_v1', 'library': binding(catalog),
        'population_count': len(choices), 'sample_count': count, 'sample': choices[:count],
        'partition': partition, 'seed': seed, 'sampling': 'task_query',
        'query': {'categories': categories, 'maximum_dimension_m': maximum_dimension},
        'code': binding(Path(__file__).with_name('materialize.py')),
        'query_implementation': binding(__file__), 'api_calls': 0,
        'physical_status': 'unmeasured', 'semantic_status': 'source_metadata_unverified',
        'automatic_replacement_of_failed_assets': False,
        'next_step': 'materialize run, then grasp/tasks init; no per-asset physics admission required',
    })
    return {'selected': count, 'matching_population': len(choices), 'plan': str(output/'plan.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--partition', choices=['development','heldout_asset'], default='development')
    parser.add_argument('--category', action='append', default=[])
    parser.add_argument('--max-dimension', type=float, default=.18)
    parser.add_argument('--seed', type=int, default=20261026)
    args = parser.parse_args()
    print(json.dumps(select(args.catalog,args.output,args.count,args.partition,args.category,args.max_dimension,args.seed),indent=2))
