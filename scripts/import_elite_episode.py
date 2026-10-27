"""Export one Elite LeRobot v3 episode, without videos or a runtime mount dependency.

Example: python scripts/import_elite_episode.py /path/to/dataset --episode 0 --output episode.npz
Requires pyarrow (already installed with the Rerun extra).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from robot_motion_workbench.adapters.elite_trajectory import RecordedTrajectory


def export_episode(root, episode, output):
    root, output = Path(root), Path(output)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info.get('codebase_version') != 'v3.0' or info.get('robot_type') != 'bi_elite_cs66_rt':
        raise ValueError('Expected a LeRobot v3 bi_elite_cs66_rt dataset')
    if output.suffix != '.npz':
        raise ValueError('Output must have the .npz extension')
    matches = []
    columns = ['episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index']
    for path in sorted((root / 'meta/episodes').glob('chunk-*/*.parquet')):
        rows = pq.read_table(path, columns=columns, filters=[('episode_index', '=', episode)]).to_pylist()
        matches.extend(rows)
    if len(matches) != 1:
        raise ValueError(f'Expected one metadata row for episode {episode}, found {len(matches)}')
    row = matches[0]
    relative = info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
    path = root / relative
    table = pq.read_table(path, columns=['action', 'timestamp', 'frame_index'],
                          filters=[('episode_index', '=', episode)]).sort_by('frame_index')
    if len(table) != row['length']:
        raise ValueError('Episode is incomplete in the referenced data file')
    meta = dict(format_version=1, robot_type=info['robot_type'], episode_index=episode,
                frame_count=len(table), fps=info['fps'], tasks=row['tasks'],
                feature_names=info['features']['action']['names'], source_column='action',
                frame='world', position_unit='m', rotation_encoding='first_two_columns',
                source_dataset=str(root.resolve()), source_data_file=relative,
                source_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
                convention_source='bi_elite_cs66_rt._cartesian_action_to_tcp_pose; '
                                  'robot_utils.quaternion_to_rotation_6d')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original float32 values exactly; convert rotations at load time.
    with output.open('xb') as destination:
        np.savez_compressed(destination, timestamps=table['timestamp'].to_numpy(),
                            frame_indices=table['frame_index'].to_numpy(),
                            actions=np.asarray(table['action'].to_pylist(), dtype=np.float32),
                            metadata=np.array(json.dumps(meta, ensure_ascii=False)))
    for arm in ('left', 'right'):
        RecordedTrajectory(output, arm=arm)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_episode(args.dataset, args.episode, args.output), indent=2))


if __name__ == '__main__':
    main()
