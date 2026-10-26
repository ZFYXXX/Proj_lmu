# Server migration: source backup scope

Target private repository: https://github.com/ZFYXXX/Projector

This source snapshot contains the current framework, compression kernels and
bindings, training/eval/maintenance scripts, configs, tests, documentation,
dependency declarations, license and third-party source notes. Third-party
nested checkouts, model weights and raw datasets are not part of this snapshot.
The existing `.gitignore` excludes outputs, runs, caches, checkpoints, model
binaries, dataset shards and credential files. The explicit upload inventory
is additionally checked by `scripts/maintenance/github_source_snapshot.py`.

The server's git/gh transport is not authenticated; the connected GitHub app
performs a single final source-snapshot commit via Git-data APIs. The remote
snapshot is based on a local source commit but does not recreate the complete
local commit graph. The upload receipt records both commit identities. The
local branch/history and frozen execution worktrees are not rewritten.

## Not backed up by a source-only push

- Trained Qwen2.5 MLP and selector weights; model/config hash identities.
- Qwen3.5 training checkpoints, optimizer/scheduler/RNG states, and latest.json.
- Per-sample predictions, completion/failure records, traces and full manifests.
- Exact dirty source snapshots/patches used by each historical frozen worktree.
- Locally isolated Transformers/Triton sources and the editable lmms-eval tree.
- Base model and raw data files (shared inputs are read-only and belong to their
  respective owners; retain source revisions/checksums instead of uploading).

These artifacts require a separate verified offline/object-storage backup if
the server filesystem will become unavailable. No such migration is claimed
by this source-only upload. No running experiment is stopped by the backup.

## Recovery checklist

1. Clone the private source repository and choose the recorded snapshot commit.
2. Recreate Python/runtime dependencies from their pinned declarations and saved
   installation-source identities; Qwen3.5 uses the isolated Transformers5.19
   path, not the default Transformers4.57 Qwen2.5 environment.
3. Restore independently saved checkpoints, manifests, predictions and frozen
   execution source; verify their checksums before resuming.
4. Restore/repoint model and data paths without changing the pinned contents.
   Runtime contracts may intentionally reject changed source/config/data paths;
   record migrations instead of silently bypassing hash checks.
5. Run CPU and real-model smoke/cache checks before resuming full jobs.
