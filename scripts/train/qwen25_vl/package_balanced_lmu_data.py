"""Package the fixed 2,560-image balanced cohort for transfer to LMU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def package(source: Path, destination: Path, archive: Path | None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, destination = source.resolve(), destination.resolve()
    manifest_path = source / 'balanced_manifest.json'
    parquet_path = source / 'balanced_qa.parquet'
    if not manifest_path.is_file() or not parquet_path.is_file():
        raise FileNotFoundError('Expected balanced_manifest.json and balanced_qa.parquet in source directory')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing package: {destination}')
    manifest = json.loads(manifest_path.read_text())
    if len(manifest.get('entries', [])) != 2560 or manifest.get('selection_reads_answers') is not False:
        raise ValueError('Only the frozen answer-independent 2,560-example cohort can be packaged')
    destination.mkdir(parents=True)
    images_dir = destination / 'images'
    images_dir.mkdir()
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if len(rows) != 2560:
        raise ValueError(f'Expected 2,560 parquet rows, found {len(rows)}')
    for row in rows:
        fingerprint = row['image_fingerprint']
        original = Path(row['image']['path'])
        if not original.is_absolute():
            original = source / original
        if not original.is_file():
            raise FileNotFoundError(original)
        target = images_dir / f'{fingerprint}.png'
        if not target.exists():
            shutil.copy2(original, target)
        row['image'] = {'path': f'images/{fingerprint}.png', 'bytes': None}
    output_parquet = destination / 'balanced_qa.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output_parquet, compression='zstd')
    for entry in manifest['entries']:
        entry['path'] = 'balanced_qa.parquet'
    manifest['sources'] = [{'path': 'balanced_qa.parquet', 'sha256': sha256(output_parquet), 'split': 'train'}]
    manifest['package_note'] = 'Portable copy of the fixed P6000-selected cohort; images use bundle-relative paths.'
    dump(destination / 'balanced_manifest.json', manifest)
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    bundle = {'format': 'qwen25_balanced_training_bundle_v1', 'examples': 2560,
              'source_counts': manifest['source_counts'],
              'files': [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]}
    dump(destination / 'bundle_manifest.json', bundle)
    if archive:
        archive = archive.resolve()
        if archive.exists():
            raise FileExistsError(f'Refusing to overwrite existing archive: {archive}')
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'w:gz', compresslevel=4) as tar:
            tar.add(destination, arcname=destination.name)


def relocate(bundle: Path) -> None:
    """Make bundle-relative parquet/manifest paths absolute after extraction on LMU."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle = bundle.resolve()
    manifest_path, parquet_path = bundle / 'balanced_manifest.json', bundle / 'balanced_qa.parquet'
    if not manifest_path.is_file() or not parquet_path.is_file():
        raise FileNotFoundError('Expected balanced_manifest.json and balanced_qa.parquet in extracted bundle')
    bundle_manifest = bundle / 'bundle_manifest.json'
    if bundle_manifest.is_file():
        recorded = json.loads(bundle_manifest.read_text())
        for item in recorded.get('files', []):
            path = bundle / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise ValueError(f'Portable training bundle integrity check failed: {path}')
    manifest = json.loads(manifest_path.read_text())
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    for row in rows:
        image = Path(row['image']['path'])
        image = image if image.is_absolute() else bundle / image
        if not image.is_file():
            raise FileNotFoundError(image)
        row['image'] = {'path': str(image), 'bytes': None}
    with tempfile.NamedTemporaryFile(prefix='balanced_qa.', suffix='.parquet', dir=bundle, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
        temporary.replace(parquet_path)
    finally:
        temporary.unlink(missing_ok=True)
    for entry in manifest['entries']:
        path = Path(entry['path'])
        entry['path'] = str(path if path.is_absolute() else bundle / path)
    manifest['sources'] = [{'path': str(parquet_path), 'sha256': sha256(parquet_path), 'split': 'train'}]
    manifest['package_note'] = 'Relocated to this LMU workspace; extracted image and parquet paths are absolute.'
    dump(manifest_path, manifest)
    files = sorted(p for p in bundle.rglob('*') if p.is_file() and p != bundle_manifest)
    dump(bundle_manifest, {'format': 'qwen25_balanced_training_bundle_v1', 'examples': len(rows),
         'source_counts': manifest['source_counts'],
         'files': [{'path': str(p.relative_to(bundle)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]})
    print(json.dumps({'bundle': str(bundle), 'examples': len(rows), 'parquet_sha256': sha256(parquet_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('package')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--archive', type=Path)
    move = commands.add_parser('relocate')
    move.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'package':
        package(args.source, args.output, args.archive)
    else:
        relocate(args.bundle)


if __name__ == '__main__':
    main()
