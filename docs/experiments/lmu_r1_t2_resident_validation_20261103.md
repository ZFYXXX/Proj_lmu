# P6000 smoke validation — 2026-11-03

Only limited checks were executed. No full evaluation was launched and no LMU
job was submitted by the agent.

| Check | Outcome |
|---|---|
| R1 across the 17 selected tasks | 17/17 passed, one generated request per task |
| T2 across the 17 selected tasks | 17/17 passed, one generated request per task |
| Release-source two-chunk check, R1 and T2 on AI2D | Four chunks passed; two model loads, two model reuses |
| Resident worker across remaining cross-task smoke cells | Two loads, 24 reuses; eight earlier R1 smoke cells were reused |
| Four-process shared claims, including R1/T2 with no dense scheduling | Passed; no duplicate chunks |
| CPU targeted tests | 12 passed |
| Slurm/env/submit Bash syntax | Passed |
| Portable full-result evidence import | 14 historical cells imported; no inference performed |

The hardware checks used one P6000-host GPU with the existing BF16/eager 7B
environment. Four logical workers were tested on CPU; this is not a physical
four-A100 claim. The submitted script requests four allocated GPUs and assigns
one persistent process per GPU.

Local evidence (not included as bulk outputs in Git):

- `outputs/eval/p6000_resident_queue_smoke_20261103_r1t2_allbench_v5/`
- `outputs/eval/p6000_release_source_r1t2_smoke_20261103_v1/`
- `outputs/eval/p6000_result_bundle_check_20261103_v1/`

The release snapshot was built on `a801bbe` with only the required R1/T2,
resident-runner, reporting, task-config and test changes. The main MLP worktree
and its previously uncommitted experiments were preserved. Early failed smoke
attempts were retained; logging and native result serialization issues were
fixed before the passing checks above.
