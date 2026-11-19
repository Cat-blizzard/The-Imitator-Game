"""JSON scalar adapter around the frozen sample audit; no decision changes."""
import argparse
from pathlib import Path
import numpy as np
from . import sample_audit
from resimpair_grasp.common import write, binding


def native_scalars(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native_scalars(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_scalars(v) for v in value]
    return value


def audit(sample, campaign, output):
    original = sample_audit.write
    def serialize(path, report):
        write(path, {**native_scalars(report), 'json_serializer': binding(__file__)})
    sample_audit.write = serialize
    try:
        return sample_audit.audit(sample, campaign, output)
    finally:
        sample_audit.write = original


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['sample', 'campaign', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(native_scalars(audit(args.sample,args.campaign,args.output)),indent=2))
