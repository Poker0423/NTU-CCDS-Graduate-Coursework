"""Create a compact, auditable Task E handover without copying model checkpoints."""
import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project.resolve()
    out = root / 'results_role3'
    out.mkdir(exist_ok=True)
    manifest = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'project': str(root), 'files': []}
    files = []
    for pattern in ('runs_role3/*/config.json', 'runs_role3/*/metrics.json',
                    'runs_role3/*/log.csv', 'runs_role3/*/ood_hcm.json',
                    'runs_role3/_logs/*.log', 'logs/role3*', 'logs/environment_*.txt',
                    'results_role3/*.json', 'results_role3/*.csv', 'results_role3/*.md',
                    'results_role3/*.png', 'scripts/role3*', 'scripts/tc2_job.sbatch'):
        files.extend(p for p in root.glob(pattern) if p.is_file())
    # The manifest is rewritten below; it cannot contain a hash of itself.
    manifest_path = out / 'artifact_manifest.json'
    files = sorted(set(files) - {manifest_path})
    for path in root.glob('runs_role3/*/outputs/*.npz'):
        manifest['files'].append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                                  'archive_includes': False})
    for path in root.glob('runs_role3/*/final.pt'):
        manifest['files'].append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                                  'archive_includes': False})
    for path in files:
        manifest['files'].append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'archive_includes': True})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = root / 'role3_task_e_results.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(set(files + [manifest_path])):
            tar.add(path, arcname=path.relative_to(root))
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'included_files': len(files) + 1}))

if __name__ == '__main__':
    main()
