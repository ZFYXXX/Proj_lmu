"""Conservative question/answer-derived S1 labels; no labels enter audio features."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import re
from pathlib import Path
from .data import ROOT, load_records, write_json

EVENT_TERMS = {
    "human_action": ("footstep", "footsteps", "walking", "running", "clapping", "applause",
                     "cough", "coughing", "sneeze", "sneezing", "breathing", "snoring",
                     "laughing", "laughter", "crying", "whistling", "screaming", "cheering"),
    "animal": ("dog", "dogs", "barking", "cat", "meowing", "bird", "birds", "chirping",
               "rooster", "cow", "horse", "sheep", "goat", "insect", "crickets", "frog",
               "owl", "duck", "ducks", "wolf", "animal"),
    "machine": ("engine", "motor", "vehicle", "car", "cars", "truck", "train", "airplane",
                "helicopter", "motorcycle", "tractor", "vacuum", "blender", "drill",
                "printer", "washing machine", "machine", "siren", "alarm", "telephone"),
    "impact": ("knocking", "hammering", "hammer", "slamming", "crash", "crashing",
               "breaking glass", "gunshot", "gunfire", "explosion", "banging", "tapping"),
    "ambient": ("rain", "raining", "thunder", "wind", "ocean", "waves", "stream",
                "waterfall", "flowing water", "river", "fire", "crackling", "rustling"),
    "music": ("piano", "guitar", "violin", "trumpet", "drums", "flute", "music", "singing"),
}

def event_label(row, override=None):
    if override is not None:
        label = override.get("event_label")
        if label is not None and label not in EVENT_TERMS:
            raise ValueError(f"Unknown event taxonomy label: {label}")
        return {"event_label": label, "status": "labeled" if label else "excluded",
                "provenance": override.get("provenance", "provided_annotation"),
                "evidence": override.get("evidence", ""), "reviewed": bool(override.get("reviewed", False))}
    if re.search(r"\b(not|neither|except|absent|without)\b", row["question"].lower()):
        return {"event_label": None, "status": "negation_requires_review",
                "provenance": "qa_lexicon_v1", "evidence": row["question"], "reviewed": False}
    # Prefer the correct answer. A question can mention multiple competing sources.
    # Abstain on negation and multiple coarse classes instead of inventing a main event.
    for field in ("answer", "question"):
        value = row[field].lower()
        if re.search(r"\b(not|neither|except|absent|without|no)\b", value):
            return {"event_label": None, "status": "negation_requires_review",
                    "provenance": "qa_lexicon_v1", "evidence": value, "reviewed": False}
        matches = {label: [term for term in terms if re.search(
            r"\b" + re.escape(term) + r"\b", value)] for label, terms in EVENT_TERMS.items()}
        matches = {label: terms for label, terms in matches.items() if terms}
        if matches:
            label = next(iter(matches)) if len(matches) == 1 else None
            return {"event_label": label, "status": "labeled" if label else "ambiguous",
                    "provenance": f"qa_lexicon_v1:{field}", "evidence": matches, "reviewed": False}
    return {"event_label": None, "status": "no_explicit_source", "provenance": "qa_lexicon_v1",
            "evidence": {}, "reviewed": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-json", default=str(ROOT.parents[1] / "AudioMCQevaluator/mmau-test-mini.json"))
    parser.add_argument("--audio-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.mini_json).resolve()
    rows = load_records(path, args.audio_root or path.parent)
    annotations = [{"id": r["id"], "question": r["question"], "answer": r["answer"],
                    **event_label(r)} for r in rows if r["task"] == "sound"]
    with Path(args.output).open("x", encoding="utf-8") as handle:
        for row in annotations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"coverage": dict(Counter(r["status"] for r in annotations)),
                      "classes": dict(Counter(r["event_label"] for r in annotations))}, ensure_ascii=False))

if __name__ == "__main__":
    main()
