"""Conditions frozen before observing task scores."""

EXPERIMENTS = ("A", "B", "C", "E", "H")


def build_conditions(experiments, *, ranks=(16, 32, 64, 128, 256),
                     stages=("P_in",), ratios=(.75, .5, .25, .1),
                     codebook_sizes=(256,), mel_ranks=(4, 8, 16, 32, 64)):
    conditions = {}
    membership = {name: [] for name in experiments}

    def add(experiment, name, family, **fields):
        condition = dict(id=name, family=family, **fields)
        if name in conditions and conditions[name] != condition:
            raise ValueError(f"Conflicting condition {name}")
        conditions[name] = condition
        membership[experiment].append(name)

    for experiment in experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(experiment)
        if experiment == "A":
            for stage in stages:
                add("A", f"A_{stage}_identity", "subspace", stage=stage, method="identity", k=0)
                for k in ranks:
                    for method in ("pca", "random"):
                        add("A", f"A_{stage}_{method}_k{k}", "subspace", stage=stage, method=method, k=k)
        if experiment in {"B", "C"}:
            for ratio in ratios:
                # Shared exact per-audio budget and mean-pooling control.
                add(experiment, f"uniform_r{ratio:g}", "segment", method="uniform", ratio=ratio)
                if experiment == "C":
                    for method in ("l2", "cosine", "random", "sample"):
                        add("C", f"C_{method}_r{ratio:g}", "segment", method=method, ratio=ratio)
                    add("C", f"C_cosine_broadcast_r{ratio:g}", "segment",
                        method="cosine", ratio=ratio, broadcast=True)
                else:
                    for k in codebook_sizes:
                        for method in ("entropy", "surprisal", "low_entropy"):
                            add("B", f"B_{method}_K{k}_r{ratio:g}", "segment",
                                method=method, ratio=ratio, codebook_size=k)
        if experiment == "E":
            for k in codebook_sizes:
                for method in ("collapse", "matched_uniform", "broadcast"):
                    add("E", f"E_{method}_K{k}", "runs", method=method, codebook_size=k)
        if experiment == "H":
            add("H", "H_identity", "mel", method="identity", k=128)
            for k in mel_ranks:
                for method in ("svd", "time_matched", "frequency_matched"):
                    add("H", f"H_{method}_k{k}", "mel", method=method, k=k)
    return list(conditions.values()), membership
