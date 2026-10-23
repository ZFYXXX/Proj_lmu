"""Two-fold, audio-disjoint PCA/k-means and first-order unit prediction."""
from pathlib import Path

import numpy as np

from .common import write_json


def assign_folds(audio_paths, seed=42):
    units = sorted(set(audio_paths))
    if len(units) < 2:
        raise ValueError("B/E cross-fitting requires at least two distinct audios")
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    mapping = {name: index % 2 for index, name in enumerate(units)}
    return [mapping[name] for name in audio_paths]


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_pca(sample, dimensions, *, seed):
    """Fit centered PCA without entering SciPy's crashing LU/SVD routines."""
    import torch
    from threadpoolctl import threadpool_limits

    if not np.isfinite(sample).all():
        raise ValueError("PCA calibration features contain non-finite values")
    previous_threads = torch.get_num_threads()
    # Bound PCA's native parallelism independently of the KMeans thread budget.
    with threadpool_limits(limits=1):
        try:
            torch.set_num_threads(1)
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                # Seed only the CPU generator; do not initialize or change CUDA RNGs.
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
                x = torch.as_tensor(sample, dtype=torch.float64, device="cpu")
                mean = x.mean(0)
                centered = x - mean
                q = min(dimensions + 10, *centered.shape)
                _, _, vectors = torch.pca_lowrank(centered, q=q, center=False, niter=7)
                components = vectors[:, :dimensions].T.contiguous()
                pivots = components.abs().argmax(dim=1)
                components *= components[torch.arange(dimensions), pivots].sign()[:, None]
                return mean.float().numpy(), components.float().contiguous().numpy()
        finally:
            torch.set_num_threads(previous_threads)


def fit_models(settings, *, resume_incomplete=False):
    import scipy
    import sklearn
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_info, threadpool_limits

    root = Path(settings["output_dir"])
    cache = root / "calibration" / "features"
    output = root / "models"
    # B/E phase restarts can refit deterministic models after an interrupted fit.
    output.mkdir(exist_ok=resume_incomplete)
    # Save the loaded numeric runtime before fitting, so a native crash leaves evidence.
    write_json(output / "fit_environment.json", dict(
        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        torch=torch.__version__, threadpools=threadpool_info(), torch_build=torch.__config__.show()))
    reports = []
    folds = np.asarray(settings["folds"])
    for fold in (0, 1):
        training_ids = np.flatnonzero(folds != fold)
        validation_ids = np.flatnonzero(folds == fold)
        if not len(training_ids) or not len(validation_ids):
            raise ValueError("Both audio folds must be nonempty")
        paths = [cache / f"doc{index}.npz" for index in training_ids]
        for stage_index, stage in enumerate(settings["stages"]):
            rng = np.random.default_rng(settings["seed"] + fold * 100 + stage_index)
            # Random-priority reservoir: equal probability per captured frame;
            # peak memory is bounded by fit_max_tokens, not dataset duration.
            sampled, priorities = None, np.empty(0)
            for path in paths:
                with np.load(path, allow_pickle=False) as arrays:
                    block = np.asarray(arrays[f"hidden_{stage}"], dtype=np.float32)
                sampled = block if sampled is None else np.concatenate((sampled, block))
                priorities = np.r_[priorities, rng.random(len(block))]
                if len(priorities) > settings["fit_max_tokens"]:
                    keep = np.argpartition(priorities, settings["fit_max_tokens"] - 1)[:settings["fit_max_tokens"]]
                    sampled, priorities = sampled[keep], priorities[keep]
            sample = normalized(sampled)
            if len(sample) < max(2, max(settings["codebook_sizes"])):
                raise ValueError(f"Fold {fold}/{stage}: {len(sample)} calibration frames < K; "
                                 "increase --limit/--calibration-tokens or reduce --codebook-sizes for smoke tests")
            dimensions = min(settings["codebook_pca_dim"], sample.shape[1], len(sample) - 1)
            print(f"fit PCA held-out fold={fold} stage={stage} shape={sample.shape} "
                  f"components={dimensions} backend=torch.pca_lowrank device=cpu "
                  f"dtype=float64 pca_threads=1", flush=True)
            mean, components = fit_pca(sample, dimensions, seed=settings["seed"])
            print(f"PCA complete held-out fold={fold} stage={stage}", flush=True)
            with threadpool_limits(limits=settings["fit_threads"]):
                # Use the same float32 projection for training and later observations.
                reduced = np.ascontiguousarray((sample - mean) @ components.T)
                for k in settings["codebook_sizes"]:
                    print(f"fit KMeans held-out fold={fold} stage={stage} K={k} "
                          f"frames={len(sample)} threads={settings['fit_threads']}", flush=True)
                    km = MiniBatchKMeans(n_clusters=k, random_state=settings["seed"],
                                         batch_size=max(1024, 3 * k), n_init=3,
                                         max_iter=100, reassignment_ratio=.01).fit(reduced)
                    transitions = np.full((k, k), settings["markov_alpha"], dtype=np.float64)
                    prior = np.full(k, settings["markov_alpha"], dtype=np.float64)
                    training_frames = training_pairs = 0
                    for path in paths:
                        with np.load(path, allow_pickle=False) as arrays:
                            hidden = normalized(arrays[f"hidden_{stage}"])
                            times = arrays[f"time_{stage}"]
                            chunks = arrays[f"chunks_{stage}"]
                        codes = km.predict(np.ascontiguousarray((hidden - mean) @ components.T))
                        prior += np.bincount(codes, minlength=k)
                        valid = (chunks[1:] == chunks[:-1]) & (np.diff(times) > 0)
                        np.add.at(transitions, (codes[:-1][valid], codes[1:][valid]), 1)
                        training_frames += len(codes)
                        training_pairs += int(valid.sum())
                    transition = transitions / transitions.sum(1, keepdims=True)
                    prior /= prior.sum()
                    filename = f"fold{fold}_{stage}_K{k}.npz"
                    np.savez_compressed(output / filename, mean=mean,
                        components=components, centers=km.cluster_centers_.astype(np.float32),
                        transition=transition, prior=prior)
                    reports.append(dict(file=filename, held_out_fold=fold, stage=stage,
                        codebook_size=k, input_dimensions=sample.shape[1], pca_dimensions=dimensions,
                        fit_tokens=len(sample), training_audios=len(training_ids),
                        held_out_audios=len(validation_ids), training_frames=training_frames,
                        training_pairs=training_pairs, additive_smoothing=settings["markov_alpha"]))
    write_json(output / "metadata.json", dict(models=reports,
        pca_backend="torch.pca_lowrank", pca_device="cpu", pca_compute_dtype="float64",
        pca_threads=1, pca_oversamples=10, pca_niter=7,
        fit_threads=settings["fit_threads"],
        design="Per-audio two-fold cross-fit; no labels or held-out audio frames in fitting",
        features="unit-normalized hidden -> train-fold PCA -> per-stage k-means",
        predictor="first-order Markov, reset at native hard chunk boundaries"))


class ModelBank:
    def __init__(self, root):
        self.root = Path(root)
        self.loaded = {}

    def get(self, fold, stage, k):
        key = (fold, stage, k)
        if key not in self.loaded:
            with np.load(self.root / f"fold{fold}_{stage}_K{k}.npz", allow_pickle=False) as data:
                self.loaded[key] = {name: data[name] for name in data.files}
        return self.loaded[key]
