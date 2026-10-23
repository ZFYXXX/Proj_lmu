"""MMSU-Final silent-redundancy observation and causal pruning analysis."""

DEFAULT_LAYERS = (3, 6, 9)
ACOUSTIC_FEATURES = ("intensity", "variation", "temporal_change")
TARGET_RATIOS = (0.75, 0.50, 0.25)
SELECTORS = ("attention", "intensity", "variation", "temporal_change", "random")
CAUSAL_SELECTORS = (
    "causal_low_change_high_attention",
    "causal_low_change_low_attention",
    "causal_high_change_attention_matched",
    "causal_low_change_random",
)
