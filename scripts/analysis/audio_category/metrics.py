"""Metrics on held-out clips; event matching and edit distance have explicit semantics."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score, r2_score, average_precision_score, normalized_mutual_info_score

def correlation(a,b):
    a,b = np.asarray(a).ravel(),np.asarray(b).ravel()
    if len(a)<2 or np.std(a)<1e-10 or np.std(b)<1e-10:
        return None
    return float(np.corrcoef(a,b)[0,1])

def event_matches(reference, prediction, tolerance):
    reference,prediction = np.asarray(reference),np.asarray(prediction)
    if not len(reference) or not len(prediction):
        return {"tp":0,"fp":len(prediction),"fn":len(reference),"errors":[]}
    costs = np.abs(reference[:,None]-prediction[None,:])
    a,b = linear_sum_assignment(np.where(costs<=tolerance,costs,1e6))
    errors = costs[a,b][costs[a,b]<=tolerance]
    return {"tp":len(errors),"fp":len(prediction)-len(errors),"fn":len(reference)-len(errors),
            "errors":errors.tolist()}

def peaks_from_scores(times,scores,tolerance):
    if not len(times):
        return np.array([])
    step = np.median(np.diff(times)) if len(times)>1 else tolerance
    peaks,_ = find_peaks(np.r_[-np.inf,scores,-np.inf],height=0,
                         distance=max(1,int(tolerance/max(step,1e-6))))
    peaks = peaks[np.asarray(scores)[peaks-1]>0]
    return np.asarray(times)[peaks-1]

def edit_distance(reference,prediction):
    row = list(range(len(prediction)+1))
    for i,a in enumerate(reference,1):
        new = [i]
        for j,b in enumerate(prediction,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(a!=b)))
        row = new
    return row[-1]

def collapse(labels):
    result = []
    previous = None
    for label in labels:
        if label != previous and label != "[sil]":
            result.append(str(label))
        previous = label
    return result

def classification(y,pred):
    return {"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0))}

def regression(y,pred):
    y,pred = np.asarray(y),np.asarray(pred)
    score = float(r2_score(y,pred)) if len(y)>1 and np.var(y)>1e-12 else None
    return {"r2":score,"correlation":correlation(y,pred),
            "mae":float(np.mean(np.abs(y-pred)))}

def metrics_for(spec,items):
    y = np.concatenate([i["truth"] for i in items])
    pred = np.concatenate([i["prediction"] for i in items])
    if spec["kind"] in ("class","event"):
        result = classification(y,pred)
    else:
        result = regression(y,pred)
    result.update(n_clips=len(items),n_observations=len(y))
    if spec["kind"]=="event":
        matches = [event_matches(i["events"],peaks_from_scores(i["times"],i["scores"],spec["tolerance"]),
                                 spec["tolerance"]) for i in items]
        tp,fp,fn = (sum(m[k] for m in matches) for k in ("tp","fp","fn"))
        result.update(event_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                      true_positive=tp,false_positive=fp,false_negative=fn,
                      auprc=float(average_precision_score(y,np.concatenate([i["scores"] for i in items])))
                      if len(np.unique(y))==2 else None)
        errors = [e for m in matches for e in m["errors"]]
        result["boundary_localization_mae_ms"] = float(np.mean(errors)*1000) if errors else None
        result["event_tolerance_ms"] = spec["tolerance"]*1000
        coverage = [i["boundary_coverage"] for i in items if i.get("boundary_coverage") is not None]
        result["retained_center_boundary_coverage"] = float(np.mean(coverage)) if coverage else None
    if spec["name"] in ("M1_chroma","M1_chroma_clip"):
        result["chroma_correlation"] = correlation(y,pred)
    if spec["name"]=="M2_bpm":
        result["tempo_accuracy_4pct"] = float(np.mean(np.abs(pred-y)<=.04*y))
        result["tempo_accuracy_octave_4pct"] = float(np.mean(np.minimum.reduce(
            [np.abs(pred-y),np.abs(pred*2-y),np.abs(pred/2-y)])<=.04*y))
    if spec["name"]=="SP1_phoneme":
        errors = sum(edit_distance(i["phones"],collapse(i["prediction"])) for i in items)
        count = sum(len(i["phones"]) for i in items)
        result["per"] = errors/count if count else None
        result["reference_phone_count"] = count
        result["prediction_nmi"] = float(normalized_mutual_info_score(y,pred))
        a = [i["cca_x"] for i in items if "cca_x" in i]
        b = [i["cca_y"] for i in items if "cca_y" in i]
        result["heldout_cca_correlation"] = correlation(np.concatenate(a),np.concatenate(b)) if a else None
        voiced = y!="[sil]"
        result["nonsilence_accuracy"] = float(np.mean(pred[voiced]==y[voiced])) if voiced.any() else None
    return result
