"""Native analysis lifecycle for the pinned 7B fixed-budget adapter."""
import hashlib
import json
from pathlib import Path
import time

from mllm_comp.analysis.common.anchor7b_runtime import (
    anchor_base, compact_anchor, implementation_hash, load_adapter, validate_identity, prepare_state, explicit_generate, file_sha,
)
from mllm_comp.analysis.common.compression_training_check import evaluate_state
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession


class Anchor7BEvaluation:
    name="anchor7b_evaluation"
    version="1.0.0"
    requires_teacher_forcing=True

    def validate_config(self, config):
        if config.get("stage") not in {"smoke","development","confirmation"}:
            raise ValueError("Explicit held-out phase required")
        for key in ["artifact_root","cohort_path","checkpoint_path","model_identity_path"]:
            if not config.get(key):
                raise ValueError(f"Missing {key}")
        if config.get("save_raw",False):
            raise ValueError("Do not save full hidden states")

    def open(self, *, backbone, runtime, config, phase, output_dir):
        self.validate_config(config)
        return Session(self,backbone,runtime,config,phase,output_dir)


class Session(EvidenceSession):
    def __post_init__(self):
        self.identity=json.loads(Path(self.config["model_identity_path"]).read_text())
        validate_identity(self.identity)
        if self.backbone.capabilities().name!="qwen25_vl" or Path(self.runtime.model_path_or_id).resolve()!=Path(self.identity["checkpoint"]).resolve():
            raise ValueError("Evaluation is not using the pinned 7B base")
        if self.runtime.revision!=self.identity["revision"]:
            raise ValueError("7B checkpoint revision identity mismatch")
        self.root,self.stage=Path(self.config["artifact_root"]),self.config["stage"]
        self.code_hash=implementation_hash()
        path=Path(self.config["checkpoint_path"])
        self.module=load_adapter(path,self.identity,self.runtime.device).eval()
        self.checkpoint_hash=file_sha(path/"projector.safetensors")
        self.config_hash=digest(json.dumps(self.config,sort_keys=True)+self.code_hash+self.checkpoint_hash+
            file_sha(Path(self.config["cohort_path"])))
        self.completed=[]
        Path(self.output_dir).mkdir(parents=True,exist_ok=True)

    def _process(self,sample,entry,folder):
        import torch
        started=time.perf_counter()
        fingerprint=media_fingerprint(sample["media"])
        if fingerprint!=entry["image_fingerprint"]:
            raise ValueError("7B held-out image mismatch")
        handles=self.backbone.model_adapter.structural_handles(self.runtime.model)
        bundle={"facade":self.backbone,"runtime":self.runtime,"model":self.runtime.model,"processor":self.runtime.processor,
                "llm":handles["llm"],"vision":handles["vision_encoder"],"device":self.runtime.device}
        teacher,_=prepare_teacher_forced_analysis_forward_state(backbone=self.backbone,runtime=self.runtime,
            sample=sample,probe_name=self.plugin.name)
        state=prepare_state(bundle,sample["processor_outputs"],teacher.native_model_inputs)
        base=anchor_base(state)
        ledger=folder/"conditions.jsonl"
        recorded=records(ledger,self.config_hash)
        for name in ["dense","feature_anchor_untrained","feature_anchor_adapter"]:
            key=(sample["sample_id"],name)
            if key in recorded:
                continue
            output,compact=None,None
            if name!="dense":
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    output=self.module(base) if name=="feature_anchor_adapter" else base
                compact=compact_anchor(state,output)
            max_new=16 if "chartqa" in sample["task_name"] else 32
            result=evaluate_state(bundle,state,compact,max_new=max_new,native_check=False)
            if self.stage=="smoke" and name=="dense":
                native=explicit_generate(bundle,state,max_new=max_new)
                if native!=result["generated_token_ids"]:
                    raise ValueError("7B native/manual dense generation mismatch")
                result["native_generation_identity_passed"]=True
            prediction=result["prediction"]
            stops=sample.get("generation_stops",[])
            for stop in ([stops] if isinstance(stops,str) else stops):
                if stop:
                    prediction=prediction.split(stop,1)[0].strip()
            metrics=sample["score_prediction"](prediction)
            strict=None
            if "chartqa" in sample["task_name"]:
                answers=sample["raw_doc"]["answer"]
                answers=answers if isinstance(answers,list) else [answers]
                strict=float(any(prediction.casefold()==str(a).strip().casefold() for a in answers))
            row={**result,"sample_id":sample["sample_id"],"task":sample["task_name"],"stage":self.stage,
                "image_fingerprint":fingerprint,"variant":name,"config_hash":self.config_hash,"status":"completed",
                "checkpoint":self.identity["checkpoint"],"revision":self.identity["revision"],
                "base_identity_sha256":self.identity["identity_sha256"],
                "projector_sha256":self.checkpoint_hash if name=="feature_anchor_adapter" else None,
                "prediction":prediction,"primary_task_score":primary_task_score(sample["task_name"],metrics),
                "strict_score":strict,"native_task_metrics":metrics,"repetition_penalty":1.,
                "compression_trace":None if output is None else {k:output[k] for k in ["indices","members","position_policy","partition_hash"]}}
            append_json(ledger,row)
            recorded[key]=row
        if len(recorded)!=3:
            raise ValueError("Incomplete 7B conditions")
        return {"sample_id":sample["sample_id"],"task":sample["task_name"],"stage":self.stage,
            "image_fingerprint":fingerprint,"config_hash":self.config_hash,"implementation_hash":self.code_hash,
            "condition_path":str(ledger),"condition_count":3,"elapsed_s":time.perf_counter()-started,
            "native_generation_identity_passed":recorded[(sample["sample_id"],"dense")].get("native_generation_identity_passed")}
