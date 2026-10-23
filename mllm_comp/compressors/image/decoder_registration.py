"""Construct a registered method for the existing native image decoder loop."""

from mllm_comp.compressors.image.decoder_config import resolve_decoder_method_config
from mllm_comp.compressors.image.decoder_lifecycle import CacheRepairPolicy
from mllm_comp.compressors.rules import load_rule_specs, rule_specs
from mllm_comp.core.errors import BackendIntegrationError, ConfigError
from mllm_comp.core.vocabulary import CompressionMethod, ExecutionPhase, Modality, Placement


def resolve_decoder_selection_depth(method, num_layers):
    """Validate physical positions at the public backbone integration boundary."""
    try:
        method.resolve_depth(num_layers)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(
            f"Invalid {method.algorithm} positions: {exc}"
        ) from exc


def prepare_decoder_selection(method, runtime):
    """Prepare method resources before native hooks are installed, when required.

    Position planning creates methods without loading checkpoint-bound assets.
    D²Pruner validates and loads its prior only at the execution boundary.
    """
    prepare = getattr(method, "prepare", None)
    if prepare is not None:
        try:
            prepare(runtime)
        except (OSError, TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid {method.algorithm} resources: {exc}") from exc


def create_registered_decoder_selection(*, stage, binding, contract_projection,
        backbone, execution_phase, binding_config=None):
    """Return the method, normalized config, loaded rules and recorded rule specs."""
    projection = contract_projection
    algorithm, variant = projection.algorithm, projection.variant
    if (binding.name not in projection.binding_names or binding.variant != variant.name
            or variant.algorithm != algorithm.name or stage.algorithm != algorithm.name
            or stage.variant != variant.name or stage.plugin != algorithm.name
            or binding.backbone != backbone):
        raise BackendIntegrationError("Image decoder registration requires one exact selected contract.")
    if (algorithm.method is not CompressionMethod.TOKEN_COMPRESSION
            or variant.target_modalities != frozenset({Modality.IMAGE})
            or len(binding.targets) != 1 or binding.targets[0].placement is not Placement.LLM_LEVEL):
        raise BackendIntegrationError("Registered image decoder methods require one image-only LLM target.")
    site = binding.targets[0].target_site
    if site not in {
        "llm_decoder_pre_prune_layer_image_tokens",
        "llm_decoder_pre_feather_stage_image_tokens",
        "llm_decoder_pre_ivc_prune_layer_image_tokens",
        "llm_decoder_layer_output_image_tokens",
        "llm_decoder_layer_post_attention_image_tokens",
    }:
        raise BackendIntegrationError(f"Unsupported native image decoder target {site!r}.")
    phase = ExecutionPhase(execution_phase)
    if phase not in {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE} or phase not in binding.supported_phases:
        raise BackendIntegrationError(f"Image decoder binding {binding.name!r} does not support {phase.value}.")
    if not callable(binding.strategy_factory):
        raise BackendIntegrationError(f"Image decoder binding {binding.name!r} requires strategy_factory.")
    try:
        config = resolve_decoder_method_config(
            stage=stage, binding=binding, variant=variant, binding_config=binding_config)
        specs = rule_specs(stage, supported_rules=variant.supported_rules)
        rules = load_rule_specs(specs)
        method = binding.strategy_factory(stage=stage, config=config, rules=rules, backbone=backbone)
    except (ConfigError, TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid registered image decoder method: {exc}") from exc
    if method.algorithm != algorithm.name or method.target_site != site:
        raise BackendIntegrationError("Image decoder method identity or target differs from its binding.")
    if method.cache_policy not in {CacheRepairPolicy.COMPACT_PREFIX, CacheRepairPolicy.LAYER_LOCAL}:
        raise BackendIntegrationError("Image decoder method requires an implemented native cache policy.")
    if method.position_policy not in {"sparse_source", "contiguous"}:
        raise BackendIntegrationError("Image decoder method requires sparse_source or contiguous positions.")
    for name in ("resolve_depth", "new_call"):
        if not callable(getattr(method, name, None)):
            raise BackendIntegrationError(f"Image decoder method requires {name}().")
    return method, config, rules, specs
