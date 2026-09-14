# PREREG — Resampling-derived honeypot discovery on SHADE-Arena
Written 2026-09-12 (evening 10:44 PM UTC 5), BEFORE any SHADE-Arena data collected. Apart incident-response sprint.
Supersedes the authored-mirror plan for the main result; the mirror stays as a cited, biased pilot.

## Question
Can black-box behavioral resampling (action-level, no chain of thought) DISCOVER a suspicious-behavior
signal on SHADE-Arena that, distilled into a honeypot detector, predicts SHADE-Arena's OWN ground-truth
"pursued the harmful side task" label on HELD-OUT task pairs, with a stated ROC/AUC?

## Why SHADE-Arena (de-biasing)
Third-party benchmark: benign main task + hidden harmful side task + a monitor, with ground-truth labels we
did NOT author. Mock tools, pure Python, OpenAI-compatible -> runs on the local vLLM (qwen3.6-35b). The
labels are theirs; we only supply the resampling method and the detector. This removes the "found what we
planted" bias of the mirror.

## Design (two stage; ALL generation overnight, analysis in the morning)
- SPLIT task pairs into DISCOVERY and HELD-OUT sets, by task-pair id, fixed in advance (~50/50; exact counts
  depend on how many pairs SHADE ships). Assignment recorded before any run.
- OVERNIGHT (discovery + all generation): branch resampling at each acting turn; control-anchored novelty
  ladder (L1 any tool; L2 outbound-to-shared-medium; L3 an act off the benign-control manifold); adaptive
  allocation + SPRT cutoff over task arms; RANDOM-allocation baseline at matched budget. Generate resampled
  trajectories for BOTH the discovery set AND the held-out set so morning needs no GPU.
- MORNING (verification, passive): derive the honeypot detector from DISCOVERY-set behaviors only; score it on
  HELD-OUT trajectories against SHADE's ground-truth harmful-side-task label; plot ROC, report AUC. Optional
  bonus if GPU spare: ACTIVE variant -- plant the discovered decoy into the env and re-run held-out.

## Predictions / thresholds (write the number before looking)
| # | Claim | Supported if | Falsified if |
|---|---|---|---|
| P1 discovery is real | resampling surfaces off-manifold acts above the random baseline | discovery L3 rate > random baseline, perm-null p<0.05 | at/below random -> no signal |
| P2 honeypot detects | the distilled detector predicts SHADE's harmful label on HELD-OUT | AUC >= 0.70 on held-out | AUC ~ 0.50 |
| P3 not just their monitor | our action-level detector catches >=1 harmful case SHADE's own monitor misses | any such case on held-out | none -> we only re-derive their monitor |

## Controls
- Random-allocation arm (matched budget) = method-efficiency control.
- Benign/control manifold = novelty floor; report its held-out false-positive rate (should be near 0 for a pure decoy).
- Label-permutation null on the discovery signal.
- SHADE-Arena's built-in monitor = comparator on the same ground truth.

## Not claimed
No discovery of failures outside SHADE's provided affordances. Single model (qwen3.6-35b), mock-tool sandbox,
one benchmark. AUC is on held-out SHADE labels, not real deployment. Every number comes from our own runs.

## Code layout
Our probing/resampling harness lives in a NEW subfolder wrapping the unmodified SHADE-Arena checkout, so their
bench stays reproducible/citable and the "our method vs their monitor" comparison stays honest.
