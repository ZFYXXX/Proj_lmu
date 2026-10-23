"""CPU verification using actual waveforms and a tiny real Transformers audio tower."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from .data import stage_names,write_json
from .labels import event_label
from .metrics import event_matches,edit_distance,collapse
from .observer import CategoryObserver,pool_tokens,interpolate
from .probes import group_folds,target_for,evaluate,task_specs
from .signal import intervene,weak_targets,read_audio,SAMPLE_RATE

def config():
    return {"folds":2,"alpha":1.0,"max_train_frames":24,"mlp_hidden":8,
            "seed":42,"keep_ratios":[1.0,.5],"compression_method":"uniform_drop"}

def test_event_labels_abstain_and_keep_evidence():
    assert event_label({"question":"What is heard?","answer":"A dog barking"})["event_label"]=="animal"
    assert event_label({"question":"What is heard?","answer":"A dog and a train"})["status"]=="ambiguous"
    assert event_label({"question":"Which sound is not heard?","answer":"No dog"})["event_label"] is None
    assert event_label({"question":"Which sound is absent?","answer":"A dog"})["event_label"] is None
    assert event_label({"question":"When?","answer":"After the bird"})["provenance"].endswith(":answer")
    with pytest.raises(ValueError,match="taxonomy"):
        event_label({},{"event_label":"speech_vs_music"})

@pytest.mark.parametrize("length",[28,91,100])
def test_native_capture_preserves_outputs_and_time_coordinates(length):
    import torch
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    torch.manual_seed(8)
    cfg = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=8,encoder_layers=2,encoder_attention_heads=4,
        encoder_ffn_dim=32,d_model=16,max_source_positions=64,n_window=15,output_dim=12)
    cfg._attn_implementation="eager"
    tower=Qwen2_5OmniAudioEncoder(cfg).eval()
    mel,lengths=torch.randn(8,length),torch.tensor([length])
    cnn,_=tower._get_feat_extract_output_lengths(lengths)
    with torch.inference_mode():
        expected=tower(mel,feature_lens=lengths,aftercnn_lens=cnn).last_hidden_state
    observer=CategoryObserver(tower,[0,1],sample_rate=16000,hop_length=160)
    observer.install()
    try:
        observer.begin()
        with torch.inference_mode():
            actual=tower(mel,feature_lens=lengths,aftercnn_lens=cnn).last_hidden_state
        result=observer.finish()
    finally:
        observer.close()
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    np.testing.assert_allclose(result["P_out__clip"],expected.mean(0).numpy(),atol=1e-7)
    expected_t=np.arange((length+1)//2)*.02
    np.testing.assert_allclose(result["C__t"],expected_t)
    n=len(expected_t)//2*2
    np.testing.assert_allclose(result["P_in__t"],expected_t[:n].reshape(-1,2).mean(1))
    assert not tower._forward_pre_hooks
    assert result["C__x"].dtype==np.float16

def test_token_compression_reduces_count_and_blurs_sharp_transition():
    x=np.r_[np.zeros(5),np.ones(5)][:,None].astype("float32")
    t=np.arange(10)*.02
    compact,centers=pool_tokens(x,t,.2)
    assert len(compact)==2
    restored=interpolate(compact,centers,t)
    assert 0<restored[4,0]<1
    assert not np.allclose(x,restored)
    np.testing.assert_array_equal(pool_tokens(x,t,1)[0],x)

def test_folds_keep_recording_groups_and_all_augmentations_together():
    groups=np.repeat(np.arange(8),2)
    seen=[]
    for train,test in group_folds(groups,4,42):
        assert not set(groups[train])&set(groups[test])
        seen.extend(test)
    assert sorted(seen)==list(range(16))

def test_one_to_one_boundary_matching_and_phone_edit_distance():
    result=event_matches([1,2],[.99,1.01,2.5],.05)
    assert (result["tp"],result["fp"],result["fn"])==(1,2,1)
    assert edit_distance(["AA","T"],["AA","S","T"])==1
    assert collapse(["AA","AA","[sil]","AA","T"])==["AA","AA","T"]

def tone(seconds=3):
    t=np.arange(round(seconds*SAMPLE_RATE))/SAMPLE_RATE
    return (.2*np.sin(2*np.pi*220*t)).astype("float32")

def test_signal_targets_interventions_and_float_gain(tmp_path):
    import soundfile as sf
    y=tone()
    targets,meta=weak_targets(y,"music")
    assert targets["chroma"].shape[1]==12
    assert meta["weak_labels"]
    assert np.nanmedian(targets["log_f0"])==pytest.approx(np.log2(220),abs=.1)
    gain,_=intervene(y,"gain_6db",targets,42)
    np.testing.assert_allclose(gain,y*10**.3,rtol=1e-6)
    stretched,info=intervene(y,"stretch_0.8",targets,42)
    assert len(stretched)==round(len(y)/.8)
    assert info["time_scale"]==.8
    shifted,_=intervene(y,"pitch_1",targets,42)
    assert len(shifted)==len(y)
    shuffled,_=intervene(y,"shuffle_250ms",targets,42)
    assert len(shuffled)==len(y)
    np.testing.assert_allclose(np.sort(shuffled),np.sort(y))
    for name in ("world_resynthesis","pitch_flatten","energy_normalize"):
        transformed,info=intervene(y,name,targets,42)
        assert len(transformed)==len(y) and np.isfinite(transformed).all()
    path=tmp_path/"gain.wav"
    sf.write(path,gain*4,SAMPLE_RATE,subtype="FLOAT")
    np.testing.assert_allclose(read_audio(path),gain*4)

def test_unvoiced_f0_and_missing_alignment_are_not_fabricated():
    spec=next(s for s in task_specs(["SP2"]) if s["name"]=="SP2_f0")
    record={"targets":{"time":np.array([0,.02,.04]),"log_f0":np.array([7,np.nan,8])},"labels":{}}
    values,mask=target_for(spec,record,np.array([0,.02,.04]))
    assert mask.tolist()==[True,False,True]
    spec=task_specs(["SP1"])[0]
    assert target_for(spec,record,np.array([0,.02]))==(None,None)

def test_textgrid_import_uses_real_phone_tier(tmp_path):
    from praatio import textgrid
    from .prepare_speech import read_textgrid
    tg=textgrid.Textgrid()
    tg.addTier(textgrid.IntervalTier("phones",[(0,.3,"AA1"),(.3,.8,"T")],0,1))
    tg.addTier(textgrid.IntervalTier("words",[(0,.8,"at")],0,1))
    path=tmp_path/"0001.TextGrid"
    tg.save(str(path),format="long_textgrid",includeBlankSpaces=True)
    aligned=read_textgrid(path)
    assert [r["label"] for r in aligned["phones"]]==["AA1","T"]
    assert aligned["words"][0]["label"]=="at"

def make_run(directory):
    directory.mkdir()
    stages=stage_names([0])
    experiments=["S1","S2","M1","M2","SP1","SP2"]
    plan={"schema":"intrinsic-audio-properties-v1","experiments":experiments,
          "probe_config":config(),"stages":stages,"shard_count":2,"seed":42,"interpretation":{},
          "sample_count":36,"full_test_mini":False}
    write_json(directory/"plan.json",plan)
    rows=[]
    rng=np.random.default_rng(19)
    for i in range(36):
        domain=("sound","music","speech")[i//12]
        row={"id":str(i),"doc_index":i,"task":domain,"group_id":f"/fixture/{i}.wav",
             "sub-category":"Emotion Flip Detection","prediction":"yes","accuracy":float(i%3!=0),
             "directory":f"{i:04d}","duration":1.0,"skipped_interventions":[]}
        shard=directory/f"shard-{i%2:03d}"
        clip=shard/row["directory"]
        clip.mkdir(parents=True)
        t=np.arange(50)*.02
        x=np.column_stack([np.sin(t*12),np.cos(t*12),np.full(50,i%3),rng.normal(size=(50,5))]).astype("float32")
        np.savez(clip/"clean.npz",**{f"{s}__{suffix}":value for s in stages
            for suffix,value in (("x",x),("t",t),("clip",x.mean(0)))})
        chroma=np.eye(12)[np.arange(50)%12]
        np.savez(clip/"targets.npz",time=t,log_energy=np.sin(t*12),log_f0=7+.1*np.cos(t*12),
                 onset_times=np.array([.2,.6]),beat_times=np.array([.2,.6]),
                 chroma=chroma,pitch_class=np.arange(50)%12)
        write_json(clip/"labels.json",{"duration":1.0,"event_annotation":{"event_label":("animal","machine","impact")[i%3]},
            "key":i%3,"bpm":90+i,"periodicity":.4+i/100,"words_per_second":2+i/10,
            "speech_alignment":{"phones":[{"start":0,"end":.3,"label":"AA1"},
                {"start":.3,"end":.6,"label":"T"},{"start":.6,"end":1.0,"label":"EH0"}]}})
        name={"sound":"gain_6db","music":"pitch_1","speech":"pitch_flatten"}[domain]
        np.savez(clip/f"{name}.npz",**{s:x.mean(0) for s in stages})
        row["variants"]=[{"name":name,"accuracy":float(i%2),"prediction":"changed",
                         "drift":{s:{"clip_cosine_distance":.1,"time_aligned_cosine_distance":.2} for s in stages}}]
        rows.append(row)
    write_json(directory/"selected.json",rows)
    for rank in range(2):
        shard=directory/f"shard-{rank:03d}"
        (shard/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows[rank::2]))
        write_json(shard/"complete.json",{"samples":18})
    return directory

def test_full_six_experiment_report_and_coverage(tmp_path):
    from .report import build_report
    from .data import load_run
    run=make_run(tmp_path/"synthetic")
    summary=build_report(run)
    assert set(summary["probes"])=={s["name"] for s in task_specs(summary["experiments"])}
    result=summary["probes"]["SP1_phoneme"]["C"]["linear"]
    assert result["status"]=="complete"
    assert "per" in result["ratios"]["1.0"]
    assert "chroma_correlation" in summary["probes"]["M1_chroma_clip"]["C"]["linear"]["ratios"]["1.0"]
    assert len(list((run/"report/figures").glob("*.png")))==7
    assert (run/"report/probe_predictions.jsonl").stat().st_size>100
    assert (run/"report/qa_interventions.csv").is_file()
    (run/"shard-001/complete.json").unlink()
    with pytest.raises(ValueError,match="did not complete"):
        load_run(run)

def test_native_runtime_stays_frozen_and_uses_only_mini():
    from .worker import runtime_config
    cfg=runtime_config({"seed":42,"checkpoint":"/fixture/model","output_dir":"/fixture/run",
                        "attention_backend":"eager"})
    assert cfg.eval.tasks==["mmaunew3_test_mini"]
    assert str(cfg.eval.compression_mode)=="none"

def test_audio_capacity_preserves_long_and_stretched_inputs():
    from transformers import WhisperFeatureExtractor
    from .worker import ensure_audio_capacity
    extractor=WhisperFeatureExtractor(chunk_length=1)
    y=tone(2.5)
    ensure_audio_capacity(extractor,len(y)/SAMPLE_RATE)
    features=extractor(y,sampling_rate=SAMPLE_RATE,return_attention_mask=True,return_tensors="np")
    assert features["attention_mask"].sum()==250
    assert extractor.n_samples>=len(y)
