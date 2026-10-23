"""Experimental Qwen3.8 projector-level compression smoke using MLLM-Comp ToMe."""

import argparse

import torch
import triton
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mllm_comp.compressors.image.tome import tome_merge_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(args.model_dir, device_map="auto").eval()
    image = Image.new("RGB", (224, 224), (0, 255, 0))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is this image? Answer in one word."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        image_outputs = model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            return_dict=True,
        )
        source_features = image_outputs.pooler_output[0]
        source_count = int(source_features.shape[0])
        original_position_ids, _ = model.model.get_rope_index(
            input_ids=inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get("attention_mask"),
        )
        baseline_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            position_ids=original_position_ids,
            mm_encoder_outputs={"image": image_outputs},
            use_cache=False,
        )
        baseline_next = processor.decode(
            baseline_output.logits[:, -1].argmax(dim=-1),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        merge_r = source_count // 2
        compressed = tome_merge_image(source_features, source_features, r=merge_r)
        compressed_features = compressed.hidden_states
        representatives = compressed.layout_delta.result_layout["representative_indices"]
        if len(representatives) != int(compressed_features.shape[0]):
            raise RuntimeError("ToMe representatives do not match compressed features.")

        input_ids = inputs["input_ids"]
        image_token_id = int(model.config.image_token_id)
        image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
        if len(image_positions) != source_count:
            raise RuntimeError(
                f"Processor/model image-token mismatch: {len(image_positions)} placeholders, "
                f"{source_count} projected features."
            )

        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep[image_positions] = False
        representative_positions = image_positions[
            torch.as_tensor(representatives, dtype=torch.long, device=input_ids.device)
        ]
        keep[representative_positions] = True

        compressed_inputs = {
            "input_ids": input_ids[:, keep],
            "attention_mask": inputs["attention_mask"][:, keep],
            "mm_token_type_ids": inputs["mm_token_type_ids"][:, keep],
            "position_ids": original_position_ids[:, :, keep],
            "mm_encoder_outputs": {
                "image": BaseModelOutputWithPooling(pooler_output=(compressed_features,))
            },
            "use_cache": False,
        }
        output = model(**compressed_inputs)

    logits = output.logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("Compressed Qwen3.8 forward produced non-finite logits.")

    answer = processor.decode(
        logits[:, -1].argmax(dim=-1),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(
        f"compressor=tome placement=post_merger_pre_llm "
        f"image_tokens={source_count}->{compressed_features.shape[0]} "
        f"sequence={input_ids.shape[1]}->{compressed_inputs['input_ids'].shape[1]} "
        f"finite_logits=True baseline_next={baseline_next!r} compressed_next={answer!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
