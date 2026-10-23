"""Run every frozen condition for every original lmms-eval request."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .observer import AudioEncoderObserver
from .interventions import AttentionInterventions, FrozenProbe, compact_audio_request
from .metrics import jsonable


class SuiteRunner:
    def __init__(self, bridge, settings, rank):
        self.bridge, self.settings, self.rank = bridge, settings, rank
        self.plan = json.loads(Path(settings['suite_plan']).read_text())
        tower = bridge.runtime.model.thinker.audio_tower
        extractor = bridge.runtime.processor.feature_extractor
        if len(tower.layers) != self.plan['encoder_layers'] or tower.layers[0].self_attn.num_heads != self.plan['encoder_heads']:
            raise ValueError('Actual encoder differs from frozen plan')
        self.observer = AudioEncoderObserver(tower, sample_rate=extractor.sampling_rate,
            hop_length=extractor.hop_length, layers=self.plan['observer_layers'],
            query_block=settings['query_block'], example_chunk=settings['example_chunk'], retain_scores=True)
        self.observer.install()
        self.controller = AttentionInterventions(tower, step_ms=2000*extractor.hop_length/extractor.sampling_rate,
                                                observer=self.observer)
        self.controller.install()
        self.probe = FrozenProbe(self.plan['probe_weights'], self.plan['probe_targets']) if self.plan['probe_weights'] else None
        self.output = Path(settings['output_dir'])
        self.records = (self.output / f'conditions.rank{rank}.jsonl').open('x', encoding='utf-8')
        self.completed = 0

    def close(self):
        self.controller.close()
        self.observer.close()
        self.records.close()

    def _call(self, callback, call_kwargs):
        device = next(self.controller.tower.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = perf_counter()
        prediction = callback(call_kwargs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return prediction, perf_counter()-started

    def run(self, sample, kwargs, callback):
        from lmms_eval.tasks.mmsufinal.utils import mmsufinal_process_results
        doc = self.bridge._document(kwargs['task'], kwargs['split'], kwargs['doc_id'])
        index = int(kwargs['doc_id'])
        seed = self.plan['seed'] + index
        actual_generation, _ = self.bridge._generation_kwargs(kwargs['request_generation'])
        generation_config = self.bridge.runtime.model.thinker.generation_config
        if actual_generation.get('thinker_do_sample', generation_config.do_sample) or actual_generation.get(
                'thinker_num_beams', generation_config.num_beams) != 1:
            raise ValueError('Paired suite requires deterministic generation')
        metadata = {k: doc.get(k) for k in ('id', 'audio_path', 'question_type', 'sub_category', 'task_name', 'dataset_name')}
        metadata.update(doc_index=index, task=kwargs['task'], split=kwargs['split'], rank=self.rank)
        example = index in self.settings['example_doc_ids']
        self.controller.begin(seed=seed)
        self.observer.begin(example=example, check_av=self.completed < self.settings['check_av_samples'], seed=seed)
        from .kernel_benchmark import capture_qkv, benchmark_reference
        benchmark_enabled = (self.completed < self.plan['kernel_benchmark_samples']
                             and any(c['family'] == 'range' for c in self.plan['conditions']))
        with capture_qkv(self.controller.tower, self.plan['layers'][0], benchmark_enabled) as benchmark_values:
            prediction, elapsed = self._call(callback, kwargs)
        measurements, arrays = self.observer.finish()
        self.controller.finish()
        original_score = mmsufinal_process_results(doc, [prediction])['accuracy']['score']
        reference = dict(metadata, prediction=prediction, accuracy=original_score, mode='suite',
                         diagnostic_generation_seconds=elapsed, **measurements)
        if self.probe:
            reference['probe'] = self.probe.read(self.controller.last_hidden, doc['id'])
        if example:
            name = f'example.rank{self.rank}.doc{index}.npz'
            np.savez_compressed(self.output / name, **arrays)
            reference['example_file'] = name
        if benchmark_enabled:
            windows = [c['window_ms'] for c in self.plan['conditions'] if c['family'] == 'range']
            reference['kernel_benchmark'] = benchmark_reference(benchmark_values, self.controller.bounds,
                self.controller.step_ms, windows)
        benchmark_values.clear()
        for condition in self.plan['conditions']:
            self.controller.begin(condition, seed=seed)
            if condition['family'] == 'input':
                self.observer.begin(example=False, check_av=False, seed=seed)
            # Only the reference prediction enters the native evaluator's cache/results.
            call_kwargs = {**kwargs, 'cache_hook': None}
            with compact_audio_request(self.bridge, self.controller):
                answer, seconds = self._call(callback, call_kwargs)
            if condition['family'] == 'input':
                self.observer.finish()
            details = self.controller.finish()
            score = mmsufinal_process_results(doc, [answer])['accuracy']['score']
            row = dict(id=doc['id'], doc_index=index, condition_id=condition['id'], prediction=answer,
                       diagnostic_generation_seconds=seconds, intervention=details)
            if condition['family'] == 'input':
                row.update(accuracy=None, original_label_agreement=score, labels_preserved=False)
            else:
                row.update(accuracy=score, labels_preserved=True)
            if self.probe:
                row['probe'] = self.probe.read(self.controller.last_hidden, doc['id'])
                if condition['family']=='input':
                    row['probe']['squared_error'] = [None]*len(row['probe']['attributes'])
                    row['probe']['targets_preserved'] = False
            self.records.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False)+'\n')
            self.records.flush()
        self.completed += 1
        return prediction, reference
