"""Fixed-four LLaMA-VID Qwen3 inference with weighted DeepStack replay."""

from pathlib import Path
from mllm_comp.compressors.video.config import VIDEO_REFERENCE_REVISION
from scripts.diagnostics import validate_qwen3_vl_video_compression_checkpoint as common
from scripts.diagnostics import (
    validate_qwen25_vl_llamavid_video_compression_checkpoint as llama,
)


def _profile():
    return common.Qwen3VideoDiagnosticProfile(
        algorithm="llamavid",
        variant="llamavid.video",
        binding="llamavid.video.qwen3_vl",
        canonical_config=llama.CANONICAL_LLAMAVID_CONFIG,
        stage_factory=lambda: llama.stage("qwen3_vl"),
        conversation_factory=lambda prompt, video, args: common._conversation(
            prompt=prompt,
            video=video,
            fps=args.video_fps,
            max_pixels=args.video_max_pixels,
        ),
        sample_validator=llama.validate_sample,
        preprocessing_record=lambda args: {
            "fps": args.video_fps,
            "max_pixels": args.video_max_pixels,
            "image_patch_size": 16,
            "enable_thinking": False,
        },
        projector_label="LLaMA-VID",
        reference_revision=VIDEO_REFERENCE_REVISION,
        auxiliary_checkpoints=(("bert_checkpoint", "llamavid_bert_path"),),
    )


def main(argv=None):
    parser = common._parser()
    parser.add_argument("--bert_checkpoint", type=Path, required=True)
    parser.set_defaults(estimated_peak_mib=32768)
    return common._execute(parser.parse_args(argv), profile=_profile())


if __name__ == "__main__":
    raise SystemExit(main())
