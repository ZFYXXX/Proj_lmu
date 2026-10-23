"""Install process-local observers, then hand control to the original evaluator."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import runpy


def main():
    import numpy as np
    import torch
    import transformers
    from mllm_comp.integrations.lmms_eval import plugin
    from .metrics import jsonable
    from .observer import AudioEncoderObserver

    settings = json.loads(Path(os.environ['AUDIO_SINK_OBSERVATION_CONFIG']).read_text())
    output_dir = Path(settings['output_dir'])
    rank = int(os.environ.get('RANK', '0'))
    sessions = []
    original = plugin.LMMSEvalRequestBridge
    if transformers.__version__ != '4.57.3':
        raise RuntimeError(f'Observer targets Transformers 4.57.3; found {transformers.__version__}')

    class ObservationBridge(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tower = self.runtime.model.thinker.audio_tower
            extractor = self.runtime.processor.feature_extractor
            self.completed = 0
            self.observer = None
            self.suite = None
            self.records_file = (output_dir / f'samples.rank{rank}.jsonl').open('x', encoding='utf-8')
            sessions.append(self)
            if settings['mode'] == 'observe':
                self.observer = AudioEncoderObserver(tower,
                    sample_rate=extractor.sampling_rate, hop_length=extractor.hop_length,
                    layers=settings['layers'], query_block=settings['query_block'],
                    example_chunk=settings['example_chunk'],
                    virtual_keep_ratio=settings['virtual_keep_ratio'], segments=settings['segments'])
                self.observer.install()
            if settings['mode'] == 'suite':
                from .suite_runner import SuiteRunner
                self.suite = SuiteRunner(self, settings, rank)
            active_observer = self.observer if self.observer is not None else (self.suite.observer if self.suite else None)
            observed_layers = active_observer.layers if active_observer is not None else []
            (output_dir / f'runtime.rank{rank}.json').write_text(json.dumps({
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'audio_source': inspect.getfile(type(tower)), 'audio_layers': len(tower.layers),
                'heads': tower.layers[0].self_attn.num_heads, 'n_window': tower.n_window,
                'audio_backend': tower.config._attn_implementation, 'dtype': str(tower.dtype),
                'sampling_rate': extractor.sampling_rate, 'hop_length': extractor.hop_length,
                'compression_mode': 'none', 'mode': settings['mode'],
                'observed_layers': observed_layers,
                'observed_head_ids': list(range(tower.layers[0].self_attn.num_heads)) if observed_layers else [],
                'observed_layer_head_pairs': len(observed_layers) * tower.layers[0].self_attn.num_heads,
            }, indent=2), encoding='utf-8')

        def _generate_sample(self, sample, **kwargs):
            if self.suite:
                prediction, record = self.suite.run(sample, kwargs,
                    lambda call_kwargs: super(ObservationBridge, self)._generate_sample(sample, **call_kwargs))
                self.records_file.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + '\n')
                self.records_file.flush()
                return prediction
            doc = self._document(kwargs['task'], kwargs['split'], kwargs['doc_id'])
            doc_index = int(kwargs['doc_id'])
            example = doc_index in settings['example_doc_ids']
            if self.observer:
                self.observer.begin(example=example,
                    check_av=self.completed < settings['check_av_samples'],
                    seed=settings['seed'] + doc_index)
            # The original bridge owns processor, model.generate, stop handling and caches.
            prediction = super()._generate_sample(sample, **kwargs)
            record = {'id': doc['id'], 'doc_index': doc_index, 'audio_path': doc['audio_path'],
                      'task': kwargs['task'], 'split': kwargs['split'], 'rank': rank,
                      'question_type': doc['question_type'], 'sub_category': doc.get('sub_category'),
                      'task_name': doc.get('task_name'), 'prediction': prediction,
                      'mode': settings['mode']}
            # Reuse the exact task scorer; no substitute answer parser.
            from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
            record['accuracy'] = mmsufinal_process_results(doc, [prediction])['accuracy']['score']
            if self.observer:
                measurements, arrays = self.observer.finish()
                record.update(measurements)
                if example:
                    name = f'example.rank{rank}.doc{doc_index}.npz'
                    np.savez_compressed(output_dir / name, **arrays)
                    record['example_file'] = name
            self.records_file.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + '\n')
            self.records_file.flush()
            self.completed += 1
            return prediction

    plugin.LMMSEvalRequestBridge = ObservationBridge
    try:
        runpy.run_module('lmms_eval', run_name='__main__')
    finally:
        plugin.LMMSEvalRequestBridge = original
        for session in sessions:
            if session.suite:
                session.suite.close()
            if session.observer:
                session.observer.close()
            session.records_file.close()


if __name__ == '__main__':
    main()
