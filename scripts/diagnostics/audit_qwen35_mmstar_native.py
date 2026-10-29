"""Compare the frozen platform's MMStar inputs and generation with native HF."""
import argparse
import json
from io import BytesIO
from pathlib import Path
import sys

import pyarrow.parquet as pq
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

SOURCE = Path('/home/ZhouFengyu/fastv-repro/worktrees/projector-extra-benchmarks-20261027-v1')
sys.path.insert(0, str(SOURCE))
from mllm_comp.analysis.common.qwen35_transfer import load_bundle
from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_5 import Qwen3_5Sample
from mllm_comp.core.serialization import atomic_write_json

BASE = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
RUN = BASE / 'outputs/eval/qwen35_vl/mmstar_protocol_check_20261027_v2'
DATA = Path('/home/ZhouFengyu/fastv-repro/datasets/MMStar/mmstar.parquet')
IDENTITY = BASE / 'outputs/eval/qwen35_vl/transfer_k128_20261026_v1/model_identity.json'
IDS = (0, 260, 530, 800, 1030, 1290)
SUFFIX = '\nAnswer with exactly one letter: A, B, C, or D. Do not explain.'


def main(output):
    if output.exists():
        raise ValueError('Refuse to overwrite previous audit')
    identity = json.loads(IDENTITY.read_text())
    bundle = load_bundle(identity, 'cuda:0')
    model, processor, facade = bundle['model'], bundle['processor'], bundle['facade']
    runtime = BackboneRuntime(model=model, processor=processor, model_path_or_id=identity['checkpoint'],
                              device=torch.device('cuda:0'), dtype=torch.bfloat16, revision=identity['revision'])
    reference = json.loads((RUN/'eval/full/dense/complete.json').read_text())
    rows = {r['doc_id']:r for r in map(json.loads, Path(reference['samples']).read_text().splitlines())}
    table = pq.read_table(DATA, columns=['image', 'question', 'answer', 'category'])
    records=[]
    for index in IDS:
        doc = table.slice(index,1).to_pylist()[0]
        with Image.open(BytesIO(doc['image'])) as im:
            image = im.convert('RGB')
        messages = [
            {'role':'system','content':[{'type':'text','text':'You are a helpful assistant.'}]},
            {'role':'user','content':[{'type':'image','image':image,'min_pixels':200704,'max_pixels':1605632},
                                     {'type':'text','text':doc['question'].strip()+SUFFIX}]}]
        sample = Qwen3_5Sample(f'audit:{index}', messages, enable_thinking=False)
        platform = facade.processor_adapter.prepare_inputs(processor, sample, phase='eval')
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16,
                                                           return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            raise ValueError('MMStar should have no video')
        native = processor(text=text, images=images, videos=None, video_metadata=None,
                           return_tensors='pt', padding=True, do_resize=False, **video_kwargs)
        if set(native) != set(platform.model_inputs):
            raise ValueError(f'Input keys differ: {set(native) ^ set(platform.model_inputs)}')
        for key, value in native.items():
            if not torch.equal(value, platform.model_inputs[key]):
                raise ValueError(f'Native and platform input differ for {key}, sample {index}')
        state=facade.state_builder.build(platform, raw_batch=sample, phase='eval')
        device_inputs={key:value.to('cuda:0') for key,value in native.items()}
        kwargs={'max_new_tokens':128, 'do_sample':False, 'repetition_penalty':1.0}
        with torch.no_grad():
            direct=model.generate(**device_inputs, **kwargs)
            produced=facade.runner.generate(runtime,state,processed_batch=platform,**kwargs)
        tail=direct[0,device_inputs['input_ids'].shape[1]:].tolist()
        generated=produced.text.token_ids[0]
        if torch.is_tensor(generated):generated=generated.tolist()
        if list(generated)!=tail:
            raise ValueError(f'Native and platform generated IDs differ for sample {index}')
        decoded=processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
        if decoded != rows[index]['filtered_resps'].strip():
            raise ValueError(f'Native generation differs from historical saved answer for sample {index}')
        records.append({'sample':index,'category':doc['category'],'target':doc['answer'],
                        'visual_tokens':int((device_inputs['input_ids']==model.config.image_token_id).sum()),
                        'input_keys':sorted(native),'output':decoded,'generated_tokens':len(tail),
                        'native_platform_input_and_token_ids_equal':True,
                        'saved_evaluation_answer_equal':True})
        print(f'audit {index} passed',flush=True)
    atomic_write_json(output,{'status':'complete','samples':records,
        'scope':'Six examples, identical model/processor inputs and greedy parameters; not official 79.7 protocol reproduction.'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
