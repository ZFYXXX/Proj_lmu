"""Clip-group CV, fixed-capacity readouts, frozen-probe token-pooling interventions."""
from __future__ import annotations
from collections import Counter
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from .signal import event_frame_labels, interval_labels
from .observer import interpolate, pool_tokens
from .metrics import metrics_for, correlation

def task_specs(experiments):
    specs = [
        ("S1_event","S1","sound","clip","class","event",None),
        ("S2_onset","S2","sound","frame","event","onset_times",.05),
        ("M1_chroma","M1","music","frame","regression","chroma",None),
        ("M1_pitch_class","M1","music","frame","class","pitch_class",None),
        ("M1_f0","M1","music","frame","regression","log_f0",None),
        ("M1_key","M1","music","clip","class","key",None),
        ("M1_chroma_clip","M1","music","clip","regression","chroma",None),
        ("M2_beat","M2","music","frame","event","beat_times",.07),
        ("M2_bpm","M2","music","clip","regression","bpm",None),
        ("M2_periodicity","M2","music","clip","regression","periodicity",None),
        ("SP1_phoneme","SP1","speech","frame","class","phones",None),
        ("SP2_f0","SP2","speech","frame","regression","log_f0",None),
        ("SP2_energy","SP2","speech","frame","regression","log_energy",None),
        ("SP2_rate","SP2","speech","clip","regression","words_per_second",None),
        ("SP2_stress","SP2","speech","frame","class","stress",None),
    ]
    return [dict(zip(("name","experiment","domain","level","kind","target","tolerance"),s))
            for s in specs if s[1] in experiments]

def group_folds(groups,folds,seed):
    unique = np.unique(groups)
    if len(unique)<folds:
        return []
    return [(np.flatnonzero(np.isin(groups,unique[a])),np.flatnonzero(np.isin(groups,unique[b])))
            for a,b in KFold(folds,shuffle=True,random_state=seed).split(unique)]

def target_for(spec,record,times):
    targets,meta = record["targets"],record["labels"]
    field = spec["target"]
    if spec["level"]=="clip":
        if field=="event":
            value = (meta.get("event_annotation") or {}).get("event_label")
        elif field=="chroma":
            value = targets["chroma"].mean(0)
        else:
            value = meta.get(field)
        if value is None:
            return None,None
        value = np.asarray([str(value)]) if spec["kind"]=="class" else np.asarray([value],dtype=float)
        return value,np.ones(1,dtype=bool)
    if field in ("phones","stress"):
        phones = meta.get("speech_alignment",{}).get("phones",[])
        if not phones:
            return None,None
        value = interval_labels(times,phones,stress=field=="stress")
        if field=="phones":
            value[value==""] = "[sil]"
        return value, value!=""
    if spec["kind"]=="event":
        value = event_frame_labels(times,targets[field],.04)
        return value,np.ones(len(times),dtype=bool)
    source = targets[field]
    grid = targets["time"]
    # Nearest weak-teacher frame for categorical/voiced labels: no interpolation through silence.
    ix = np.searchsorted(grid,times)
    right = np.clip(ix,0,len(grid)-1)
    left = np.clip(ix-1,0,len(grid)-1)
    ix = np.where(np.abs(grid[left]-times)<np.abs(grid[right]-times),left,right)
    value = np.asarray(source[ix])
    valid = np.isfinite(value).all(axis=1) if value.ndim==2 else np.isfinite(value)
    if spec["kind"]=="class":
        value = np.asarray([str(int(v)) if np.isfinite(v) else "" for v in value])
    return value,valid

def fit_probe(x,y,kind,config,head):
    if kind in ("class","event"):
        if len(np.unique(y))<2:
            raise ValueError("training_fold_has_fewer_than_two_classes")
        estimator = MLPClassifier(hidden_layer_sizes=(config["mlp_hidden"],),alpha=1e-3,
                    max_iter=300,random_state=config["seed"],early_stopping=False) if head=="mlp" else \
                    RidgeClassifier(alpha=config["alpha"],class_weight="balanced",solver="lsqr",tol=1e-4)
    else:
        estimator = Ridge(alpha=config["alpha"],solver="lsqr",tol=1e-4)
    return make_pipeline(StandardScaler(),estimator).fit(x,y)

def fit_cca(x,y,seed):
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    labels = np.unique(y)
    if len(labels)<2 or len(x)<4:
        return None
    ix = np.random.default_rng(seed).choice(len(x),size=min(4096,len(x)),replace=False)
    scale = make_pipeline(StandardScaler(),PCA(n_components=min(32,x.shape[1],len(ix)-1),
                                               svd_solver="randomized",random_state=seed))
    z = scale.fit_transform(x[ix])
    onehot = (y[ix,None]==labels[None,:]).astype(float)
    # Remove unrepresented classes from the fitting subsample.
    labels = labels[onehot.sum(0)>0]
    if len(labels)<2:
        return None
    onehot = (y[ix,None]==labels[None,:]).astype(float)
    cca = CCA(n_components=1,max_iter=500).fit(z,onehot)
    return scale,cca,labels

def evaluate(spec,records,stage,config,head="linear"):
    """All variants and compressed versions remain in their original clip's outer test fold."""
    data,excluded = [],[]
    for record in records:
        x,t = record["x"],record["t"]
        y,mask = target_for(spec,record,t)
        if y is None or not mask.any():
            excluded.append({"id":record["row"]["id"],"reason":"missing_or_invalid_weak_target"})
            continue
        data.append({**record,"y":y,"mask":mask})
    groups = np.asarray([d["row"]["group_id"] for d in data])
    folds = group_folds(groups,config["folds"],config["seed"])
    metadata = {"eligible_clips":len(data),"excluded":excluded,
                "feature_dim":records[0]["x"].shape[1] if records else None}
    if not folds:
        return {"status":"insufficient_data",**metadata},[],[],[]
    ratios = config["keep_ratios"]
    results = {ratio:[] for ratio in ratios}
    predictions,interventions,fold_info = [],[],[]
    for fold,(train,test) in enumerate(folds):
        xx,yy = [],[]
        for index in train:
            d = data[index]
            if spec["level"]=="clip":
                xx.append(d["clip"][None,:])
                yy.append(d["y"])
            else:
                ix = np.flatnonzero(d["mask"])
                ix = ix[np.linspace(0,len(ix)-1,min(len(ix),config["max_train_frames"]),dtype=int)]
                xx.append(d["x"][ix])
                yy.append(d["y"][ix])
        xtrain,ytrain = np.concatenate(xx),np.concatenate(yy)
        try:
            model = fit_probe(xtrain,ytrain,spec["kind"],config,head)
        except ValueError as exc:
            if str(exc)!="training_fold_has_fewer_than_two_classes":
                raise
            return {"status":"insufficient_data","reason":str(exc),**metadata},[],[],[]
        cca = fit_cca(xtrain,ytrain,config["seed"]+fold) if spec["name"]=="SP1_phoneme" else None
        fold_info.append({"fold":fold,"train_ids":[data[i]["row"]["id"] for i in train],
                          "test_ids":[data[i]["row"]["id"] for i in test],
                          "train_observations":len(ytrain),
                          "mlp_iterations":int(model[-1].n_iter_) if head=="mlp" else None,
                          "mlp_hit_iteration_limit":bool(model[-1].n_iter_>=model[-1].max_iter) if head=="mlp" else None,
                          "training_classes":np.unique(ytrain).tolist() if spec["kind"] in ("class","event") else None})
        for index in test:
            d = data[index]
            row,t,mask = d["row"],d["t"],d["mask"]
            for ratio in ratios:
                if config.get("compression_method","mean_pool")=="uniform_drop" and ratio<1:
                    keep = np.linspace(0,len(t)-1,max(1,int(np.ceil(len(t)*ratio))),dtype=int)
                    compact,centers = d["x"][keep],t[keep]
                else:
                    compact,centers = pool_tokens(d["x"],t,ratio)
                # Pooling reduces native tokens. Reconstruction supplies the same evaluation
                # timestamps to a frozen clean-trained head, rather than fitting on test targets.
                testx = (compact.mean(0)[None,:] if spec["level"]=="clip" else
                         interpolate(compact,centers,t)[mask])
                if ratio==1 and spec["level"]=="clip":
                    testx = d["clip"][None,:]
                pred = model.predict(testx)
                truth = d["y"][mask]
                item = {"truth":truth,"prediction":pred}
                if spec["level"]=="frame":
                    item["times"] = t[mask]
                if spec["kind"]=="event":
                    item["scores"] = model.decision_function(testx)
                    item["events"] = d["targets"][spec["target"]]
                    item["boundary_coverage"] = float(np.mean(np.min(
                        np.abs(item["events"][:,None]-centers[None,:]),axis=1)<=spec["tolerance"])) if len(item["events"]) else None
                if spec["name"]=="SP1_phoneme":
                    phones = d["labels"]["speech_alignment"]["phones"]
                    item["phones"] = [re.sub(r"[012]$","",v["label"]) for v in phones
                                      if v["label"].lower() not in {"sil","sp","spn","<eps>"}]
                    if cca is not None:
                        scaler,canonical,classes = cca
                        known = np.isin(truth,classes)
                        if known.any():
                            a,b = canonical.transform(scaler.transform(testx[known]),
                                                       (truth[known,None]==classes[None,:]).astype(float))
                            item.update(cca_x=a.ravel(),cca_y=b.ravel())
                results[ratio].append(item)
                predictions.append({"id":row["id"],"fold":fold,"ratio":ratio,
                                    "times":item.get("times",np.array([])).tolist(),
                                    "truth":truth.tolist(),"prediction":pred.tolist(),
                                    "scores":item["scores"].tolist() if "scores" in item else None,
                                    "native_tokens":len(t),"retained_tokens":len(centers),
                                    "unseen_test_labels":sorted(set(truth)-set(ytrain))
                                      if spec["kind"] in ("class","event") else []})
                if ratio==1 and spec["name"] in ("S1_event","M1_chroma_clip"):
                    for name,vector in d["variant_clips"].items():
                        changed = model.predict(vector[None,:])
                        result = {"id":row["id"],"fold":fold,"variant":name}
                        if spec["name"]=="S1_event" and name in ("gain_6db","noise_20db","shift_250ms"):
                            result.update(consistent=float(changed[0]==pred[0]),
                                          accuracy=float(changed[0]==truth[0]))
                            interventions.append(result)
                        if spec["name"]=="M1_chroma_clip" and name.startswith("pitch_"):
                            shift = int(name.split("_")[1])
                            result.update(chroma_equivariance_correlation=correlation(
                                np.roll(pred[0],shift),changed[0]),
                                chroma_invariance_correlation=correlation(pred[0],changed[0]))
                            interventions.append(result)
        del xtrain,ytrain,xx,yy
    metrics = {str(ratio):metrics_for(spec,items) for ratio,items in results.items()}
    clean = metrics["1.0"] if "1.0" in metrics else metrics["1"]
    for ratio,values in metrics.items():
        values["delta_from_clean"] = {key: values[key]-clean[key]
             for key in ("accuracy","macro_f1","r2","correlation","event_f1","per","chroma_correlation")
             if isinstance(values.get(key),(float,int)) and isinstance(clean.get(key),(float,int))}
    return {"status":"complete",**metadata,"ratios":metrics},predictions,interventions,fold_info
