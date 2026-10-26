"""Delete only owner-local derived Arrow caches backed by existing raw files.

Default is a dry-run manifest. --apply requires that exact reviewed manifest;
every path, ownership, source and payload is validated again before deletion.
Never traverse symlinks, raw datasets, model hubs, checkpoint/result directories.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

OWNER = Path('/home/ZhouFengyu')
PLATFORM = OWNER / 'fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main'
OUTPUTS = PLATFORM / 'outputs'
GLOBAL_CACHE = OWNER / '.cache/huggingface/datasets'


def active_cache_paths():
    """Protect owner processes' cache environments and open/mapped Arrow files.

    Read only cache-related values; never log the complete process environment.
    No process is stopped and no other user's environment is inspected.
    """
    paths = set()
    def add(raw):
        if not raw.startswith(str(OWNER) + '/'):
            return
        path = Path(raw.removesuffix(' (deleted)')).resolve()
        if path.is_relative_to(OWNER):
            paths.add(path)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for item in (proc / 'environ').read_bytes().split(b'\0'):
                key, sep, value = item.partition(b'=')
                if sep and key in {b'HF_HOME', b'HF_DATASETS_CACHE', b'XDG_CACHE_HOME', b'TORCH_HOME'}:
                    add(value.decode(errors='replace'))
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and (fields[5].endswith('.arrow') or fields[5].endswith('.arrow (deleted)')):
                    add(fields[5])
            for descriptor in (proc / 'fd').iterdir():
                try:
                    raw = os.readlink(descriptor)
                    if '/cache/' in raw or str(GLOBAL_CACHE) in raw:
                        add(raw)
                except (OSError, ProcessLookupError):
                    continue
        except (OSError, ProcessLookupError):
            continue  # Exiting or non-readable process; explicit prefixes remain protected.
    return paths


def protected_reason(folder, explicit):
    for path in {*explicit, *active_cache_paths()}:
        if folder == path or folder.is_relative_to(path) or path.is_relative_to(folder):
            return f'Active or explicitly preserved cache: {path}'
    return None


def validate_leaf(folder):
    folder = Path(folder)
    if folder.is_symlink() or folder.resolve() != folder or not folder.is_relative_to(OWNER):
        raise ValueError(f'Unsafe cache path: {folder}')
    if not (folder.is_relative_to(GLOBAL_CACHE) or
            (folder.is_relative_to(OUTPUTS) and '/cache/datasets/' in str(folder))):
        raise ValueError(f'Not an approved derived-cache leaf: {folder}')
    info_path = folder / 'dataset_info.json'
    info = json.loads(info_path.read_text())
    checksums = info.get('download_checksums') or {}
    if not checksums:
        raise ValueError('No raw-source evidence in cache metadata')
    sources = []
    for raw in checksums:
        source = Path(raw)
        if not source.is_absolute() or not source.is_file() or not os.access(source, os.R_OK):
            raise ValueError(f'Raw source is not an existing readable local file: {raw}')
        if source.is_relative_to(folder):
            raise ValueError('Raw source cannot be inside the cache being removed')
        sources.append({'path': str(source), 'bytes': source.stat().st_size})
    paths = [folder, *folder.rglob('*')]
    allocated, logical, count = 0, 0, 0
    for path in paths:
        stat = path.lstat()
        if path.is_symlink() or stat.st_uid != os.getuid():
            raise ValueError(f'Symlink or foreign-owned entry: {path}')
        if path.is_file():
            if path.suffix not in {'.arrow', '.json', '.lock', '.tmp'}:
                raise ValueError(f'Unexpected/non-cache payload: {path}')
            allocated += stat.st_blocks * 512
            logical += stat.st_size
            count += 1
        elif not path.is_dir():
            raise ValueError(f'Nonregular cache entry: {path}')
    return {'path': str(folder), 'files': count, 'allocated_bytes': allocated,
            'logical_bytes': logical, 'raw_sources_preserved': sources}


def main(args):
    report = args.manifest.resolve()
    if not report.is_relative_to(OUTPUTS / 'maintenance'):
        raise ValueError('Cleanup manifest must be in owner outputs/maintenance')
    explicit = {Path(p).resolve() for p in args.exclude_prefix}
    if any(not p.is_relative_to(OWNER) or p == OWNER for p in explicit):
        raise ValueError('Preserved prefixes must be specific owner-local paths')
    if not args.apply:
        if report.exists():
            raise ValueError('Preserve previous audit manifests; use a new filename')
        candidates = set()
        if GLOBAL_CACHE.exists():
            candidates.update(p.parent for p in GLOBAL_CACHE.rglob('dataset_info.json'))
        # Restrict recursive inspection to derived dataset-cache directories.
        for current, dirs, _ in os.walk(OUTPUTS, followlinks=False):
            path = Path(current)
            dirs[:] = [name for name in dirs if name not in {'checkpoints', 'local_sources', 'cache'}
                       and not (path / name).is_symlink()]
            cache = path / 'cache/datasets'
            if cache.is_dir() and not cache.is_symlink():
                candidates.update(p.parent for p in cache.rglob('dataset_info.json'))
        selected, skipped = [], []
        for folder in sorted(candidates):
            try:
                reason = protected_reason(folder, explicit)
                if reason:
                    skipped.append({'path': str(folder), 'reason': reason})
                    continue
                selected.append(validate_leaf(folder))
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                skipped.append({'path': str(folder), 'reason': str(exc)})
        result = {'status': 'dry_run', 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'scope': 'Only derived local-backed Arrow leaves; raw data/model/checkpoint/result/log files excluded',
                  'targets': selected, 'skipped': skipped,
                  'preserved_prefixes': sorted(map(str, explicit)),
                  'planned_bytes': sum(r['allocated_bytes'] for r in selected)}
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'targets': len(selected), 'skipped': len(skipped),
                          'planned_GiB': result['planned_bytes'] / 1024**3, 'manifest': str(report)}))
        return
    result = json.loads(report.read_text())
    if result['status'] != 'dry_run':
        raise ValueError('This manifest was already applied or changed')
    explicit.update(Path(p) for p in result.get('preserved_prefixes', []))
    # Validate all before removing any, then once again immediately per target.
    for entry in result['targets']:
        if reason := protected_reason(Path(entry['path']), explicit):
            raise ValueError(reason)
        if validate_leaf(entry['path']) != entry:
            raise ValueError(f'Cache changed after review: {entry["path"]}')
    result.update(status='applying', removed=[])
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for entry in result['targets']:
        if reason := protected_reason(Path(entry['path']), explicit):
            raise ValueError(reason)
        if validate_leaf(entry['path']) != entry:
            raise ValueError(f'Cache changed before deletion: {entry["path"]}')
        shutil.rmtree(Path(entry['path']))
        result['removed'].append(entry['path'])
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    result.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                  removed_bytes=result['planned_bytes'])
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'complete', 'removed_directories': len(result['removed']),
                      'removed_GiB': result['removed_bytes'] / 1024**3}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--exclude-prefix', action='append', default=[],
                        help='Preserve this owner-local directory and all caches below it')
    main(parser.parse_args())
