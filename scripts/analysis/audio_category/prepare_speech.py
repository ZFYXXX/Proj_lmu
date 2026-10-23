"""Create weak phoneme/word intervals with faster-whisper and MFA 3.x."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from .data import ROOT, load_records, validate_speech, write_json

def read_textgrid(path):
    from praatio import textgrid
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    result = {}
    for kind in ("phones", "words"):
        names = [n for n in tg.tierNames if n == kind or n.endswith(" - " + kind)]
        if len(names) != 1:
            raise ValueError(f"Expected one {kind} tier in {path}, found {names}")
        result[kind] = [{"start": float(a), "end": float(b), "label": value}
                        for a,b,value in tg.getTier(names[0]).entries
                        if value.strip() and value.lower() not in {"sil","sp","spn","<eps>"}]
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-json", default=str(ROOT.parents[1]/"AudioMCQevaluator/mmau-test-mini.json"))
    parser.add_argument("--audio-root")
    parser.add_argument("--output-dir", required=True, help="New directory for corpus, alignments and JSONL")
    parser.add_argument("--asr-model", default="small.en", help="faster-whisper model name or local path")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--dictionary", default="english_us_arpa", help="MFA 3.x dictionary path/name")
    parser.add_argument("--acoustic-model", default="english_us_arpa")
    parser.add_argument("--mfa-bin", default="mfa")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--transcripts-jsonl", help="Optional reviewed {id, transcript} rows; skip ASR")
    parser.add_argument("--textgrid-dir", help="Import existing <doc_index:04d>.TextGrid files; skip ASR/MFA")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from .signal import SAMPLE_RATE, read_audio
    import soundfile as sf
    path = Path(args.mini_json).resolve()
    rows = [r for r in load_records(path, args.audio_root or path.parent) if r["task"]=="speech"]
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.output_dir).resolve()
    if not args.textgrid_dir and not shutil.which(args.mfa_bin):
        parser.error("MFA 3.x executable missing; install in a separate conda environment or use --textgrid-dir")
    output.mkdir(parents=True, exist_ok=False)
    transcripts = {}
    if args.transcripts_jsonl:
        transcripts = {r["id"]: r["transcript"] for r in
                       map(json.loads, Path(args.transcripts_jsonl).read_text().splitlines())}
    asr = None
    if not args.textgrid_dir and not args.transcripts_jsonl:
        from faster_whisper import WhisperModel
        asr = WhisperModel(args.asr_model, device=args.device, device_index=args.device_index,
                           compute_type="int8" if args.device=="cpu" else "float16")
    transcript_rows = []
    if not args.textgrid_dir:
        for row in rows:
            y = read_audio(row["audio_path"])
            if asr is not None:
                segments, info = asr.transcribe(y, language=args.language, beam_size=5,
                                                 vad_filter=True, condition_on_previous_text=False)
                transcript = " ".join(segment.text.strip() for segment in segments).strip()
            else:
                transcript = transcripts.get(row["id"], "").strip()
            transcript_rows.append({"id":row["id"], "doc_index":row["doc_index"], "transcript":transcript})
            if not transcript:
                continue
            # One directory per clip avoids asserting that different clips share a speaker.
            corpus = output/"corpus"/f"{row['doc_index']:04d}"
            corpus.mkdir(parents=True)
            sf.write(corpus/f"{row['doc_index']:04d}.wav", y, SAMPLE_RATE, subtype="PCM_16")
            (corpus/f"{row['doc_index']:04d}.lab").write_text(transcript, encoding="utf-8")
        write_json(output/"transcripts.json", transcript_rows)
        if not any(r["transcript"] for r in transcript_rows):
            raise ValueError("No nonempty transcripts to align")
        mfa_bin = str(Path(shutil.which(args.mfa_bin)).resolve())
        mfa_env = {**os.environ,"PATH":str(Path(mfa_bin).parent)+os.pathsep+os.environ.get("PATH","")}
        subprocess.run([mfa_bin, "align", str(output/"corpus"), args.dictionary,
                        args.acoustic_model, str(output/"aligned"), "--num_jobs", str(args.jobs),
                        "--uses_speaker_adaptation", "false"], check=True, env=mfa_env)
    aligned = Path(args.textgrid_dir).resolve() if args.textgrid_dir else output/"aligned"
    grids = {}
    for path in aligned.rglob("*.TextGrid"):
        if path.stem in grids:
            raise ValueError(f"Duplicate TextGrid name: {path.stem}")
        grids[path.stem] = path
    covered = 0
    with (output/"speech_alignment.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            name = f"{row['doc_index']:04d}"
            annotation = {"id": row["id"], "phones": [], "words": [],
                          "provenance": "provided_TextGrid" if args.textgrid_dir else
                          (f"provided_transcript+MFA:{args.acoustic_model}" if args.transcripts_jsonl else
                           f"faster-whisper:{args.asr_model}+MFA:{args.acoustic_model}"),
                          "language": args.language, "weak_labels": True}
            if name in grids:
                annotation.update(read_textgrid(grids[name]))
                validate_speech(annotation)
                covered += bool(annotation["phones"])
            annotation["status"] = "aligned" if annotation["phones"] else "missing_alignment"
            handle.write(json.dumps(annotation, ensure_ascii=False)+"\n")
    write_json(output/"preparation.json", {"samples":len(rows), "aligned":covered,
               "dictionary":args.dictionary, "acoustic_model":args.acoustic_model,
               "language":args.language, "phone_set":"MFA acoustic model inventory; stress suffix stripped for SP1",
               "warning":"ASR/forced-alignment weak labels; inspect transcript and alignment quality"})
    print(f"Aligned {covered}/{len(rows)} clips: {output/'speech_alignment.jsonl'}")

if __name__ == "__main__":
    main()
