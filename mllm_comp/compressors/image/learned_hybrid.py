"""Small trainable post-merger compressors for controlled Qwen2.5-VL pilots.

This is a native rectangular-grid adaptation, not an official reproduction.
ReConV supplies summary-conditioned reconstruction/routing; Braco motivates
the optional DCT -> coarse IDCT initialization. See the experiment source audit.
"""

from dataclasses import asdict, dataclass
import math


VARIANTS = ("pool_adapter", "summary_random", "summary_residual", "spectral_residual")


@dataclass(frozen=True)
class LearnedHybridSpec:
    hidden_size: int
    variant: str
    target_tokens: int = 128
    summary_tokens: int = 64
    width: int = 256
    heads: int = 8
    reconstruction_anchors: int = 128
    temperature: float = .1
    ste_temperature: float = .2

    def __post_init__(self):
        if self.variant not in VARIANTS or self.hidden_size < 1 or not 0 < self.summary_tokens < self.target_tokens:
            raise ValueError("Invalid hybrid variant/dimension/token split.")
        if self.width < 1 or self.heads < 1 or self.width % self.heads or self.temperature <= 0 or self.ste_temperature <= 0:
            raise ValueError("Invalid attention/reconstruction configuration.")


def grid_shape(height, width, count):
    choices = [(h, count // h) for h in range(1, min(height, count) + 1)
               if count % h == 0 and count // h <= width]
    if not choices:
        raise ValueError("No nonempty rectangular exact-budget partition.")
    return min(choices, key=lambda x: (abs(math.log((x[0] / x[1]) / (height / width))), x))


def spatial_groups(height, width, count):
    rows, cols = grid_shape(height, width, count)
    groups = []
    for r in range(rows):
        for c in range(cols):
            group = [y * width + x for y in range(r * height // rows, (r + 1) * height // rows)
                     for x in range(c * width // cols, (c + 1) * width // cols)]
            if not group:
                raise ValueError("Empty spatial cell.")
            groups.append(group)
    return groups, (rows, cols)


def create_learned_hybrid(spec):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from mllm_comp.analysis.common.spectral_residual import dct_matrix, spatial_dct, inverse_dct

    class LearnedHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            d, w = spec.hidden_size, spec.width
            self.adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w), nn.GELU(), nn.Linear(w, d))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
            if spec.variant != "pool_adapter":
                self.visual_norm = nn.LayerNorm(d)
                self.query = nn.Linear(d, w)
                self.key = nn.Linear(d, w)
                self.value = nn.Linear(d, w)
                self.slots = nn.Parameter(torch.randn(spec.summary_tokens, w) * .01)
                self.position = nn.Sequential(nn.Linear(2, w), nn.GELU(), nn.Linear(w, w))
                self.summary_up = nn.Linear(w, d)
                nn.init.zeros_(self.summary_up.weight)
                nn.init.zeros_(self.summary_up.bias)
                self.reconstruction_down = nn.Linear(d, w)
                self.decoder = nn.Sequential(nn.Linear(w * 2, w), nn.GELU(), nn.Linear(w, d))

        def forward(self, features, *, height, width, random_seed=42):
            if features.ndim != 2 or features.shape != (height * width, spec.hidden_size) or len(features) < spec.target_tokens:
                raise ValueError("Finite single-image rectangular grid with N>=K required.")
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Nonfinite original visual features.")
            x = features.detach().float()
            count = spec.target_tokens if spec.variant == "pool_adapter" else spec.summary_tokens
            groups, coarse = spatial_groups(height, width, count)
            means = torch.stack([x[group].mean(0) for group in groups])
            representatives = []
            for group in groups:
                coordinates = [(i // width, i % width) for i in group]
                cy = sum(p[0] for p in coordinates) / len(group)
                cx = sum(p[1] for p in coordinates) / len(group)
                representatives.append(min(group, key=lambda i: ((i // width - cy) ** 2 + (i % width - cx) ** 2, i)))
            rec_loss = x.new_zeros(())
            selection_scores = None
            details = []
            if spec.variant == "pool_adapter":
                output = means
            else:
                base = means
                if spec.variant == "spectral_residual":
                    # DCT coefficients and spatial means are first put on the
                    # SAME coarse spatial lattice, not added in different bases.
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        coefficients, _, _ = spatial_dct(x, height, width)
                        rh, rw = coarse
                        low = coefficients[:rh, :rw]
                        frequency_grid = inverse_dct(low, dct_matrix(rh, device=x.device), dct_matrix(rw, device=x.device))
                        # Preserve a constant field's native amplitude at init.
                        frequency_grid *= math.sqrt(count / len(x))
                        base = .5 * means + .5 * frequency_grid
                all_ids = torch.arange(len(x), device=x.device)
                xy = torch.stack((all_ids.div(width, rounding_mode="floor") / max(1, height - 1),
                                  all_ids.remainder(width) / max(1, width - 1)), -1).float()
                normed = self.visual_norm(x)
                q = self.query(base) + self.slots
                k, v = self.key(normed), self.value(normed)
                h, head_dim = spec.heads, spec.width // spec.heads
                logits = torch.einsum("mhd,nhd->hmn", q.reshape(count, h, head_dim), k.reshape(len(x), h, head_dim)) / math.sqrt(head_dim)
                weights = logits.float().softmax(-1).to(v.dtype)
                context = torch.einsum("hmn,nhd->mhd", weights, v.reshape(len(x), h, head_dim)).reshape(count, spec.width)
                summaries = base + self.summary_up(context).float()
                route = logits.float().mean(0).argmax(0)
                reconstruction = self.decoder(torch.cat((self.reconstruction_down(summaries[route]), self.position(xy)), -1)).float()
                # Fractional orthogonal residual removes raw feature norm bias.
                # This differs explicitly from ReConV's unnormalized score.
                u, target = F.normalize(reconstruction, dim=-1), F.normalize(x, dim=-1)
                selection_scores = (1 - (target * u).sum(-1).square()).clamp_min(0)
                anchors = min(spec.reconstruction_anchors, len(x))
                gen = torch.Generator(device="cpu").manual_seed(int(random_seed))
                anchor_ids = torch.randperm(len(x), generator=gen)[:anchors].to(x.device)
                rec_logits = u[anchor_ids] @ target.T / spec.temperature
                rec_loss = F.cross_entropy(rec_logits, anchor_ids)
                blocked = set(representatives)
                eligible = [i for i in range(len(x)) if i not in blocked]
                amount = spec.target_tokens - count
                if spec.variant == "summary_random":
                    permutation = torch.randperm(len(eligible), generator=gen)[:amount].tolist()
                    details = [eligible[i] for i in permutation]
                else:
                    # Stable sort makes index the tie-break; gradient enters
                    # only the selected gates, not the discrete top-k itself.
                    pool = torch.tensor(eligible, device=x.device)
                    order = torch.argsort(selection_scores[pool], descending=True, stable=True)
                    details = pool[order[:amount]].tolist()
                detail_values = x[details]
                if spec.variant != "summary_random":
                    threshold = selection_scores[pool[order[amount]]].detach() if amount < len(eligible) else selection_scores.min().detach()
                    soft = torch.sigmoid((selection_scores[details] - threshold) / spec.ste_temperature)
                    gate = 1 + soft - soft.detach()
                    detail_values = detail_values * gate[:, None]
                output = torch.cat((summaries, detail_values), 0)
            indices = representatives + details
            permutation = sorted(range(len(indices)), key=indices.__getitem__)
            output = output[permutation]
            output = output + self.adapter(output).float()
            retained = [indices[i] for i in permutation]
            if len(retained) != len(set(retained)) or len(retained) != spec.target_tokens or not bool(torch.isfinite(output).all()):
                raise ValueError("Hybrid exact-budget, uniqueness or finite-value failure.")
            return {"tokens": output.to(features.dtype), "indices": retained, "reconstruction_loss": rec_loss,
                    "summary_indices": representatives, "detail_indices": details, "members": groups,
                    "score_mean": float(selection_scores.detach().mean()) if selection_scores is not None else None,
                    "score_std": float(selection_scores.detach().std(unbiased=False)) if selection_scores is not None else None,
                    "position_policy": "original_representative_and_original_detail", "coarse_grid": list(coarse),
                    "gate_gradient": "biased_STE_on_selected_only" if spec.variant.endswith("residual") else "none"}

        def config_dict(self):
            return asdict(self.spec)

    return LearnedHybrid()
