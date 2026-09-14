# sirens_discovery — Sirens for Agents: resampling-derived, action-level honeypot discovery

Minimum artifact to recreate our Apart AI Incident Response sprint work (Sep 2026).

**One line.** We probe a black-box agent at the ACTION level (no chain-of-thought), resampling its behavior on a
real third-party sabotage benchmark (SHADE-Arena) to discover covert-behavior "honeypot" affordances, then
distill a cheap tripwire. **Honest headline finding: the bottleneck is discovery, not detection, and it is
compute-bound.**

## What is honestly here (and not)
- The harness runs on SHADE and a control-anchored action detector separated completed harm from benign (2/2,
  0/5) and out-flagged SHADE's own monitor once — but this is STRUCTURAL (harm needs a dedicated tool) and n=2.
- The adaptive, resource-efficient engine (subset-simulation ladder + SPRT + allocation) is SPECIFIED and
  released but NOT yet validated: a full episode did not finish in 3h on local hardware.
- Read `docs/STORY.md` and `docs/VERIFICATION.md` FIRST — they state exactly what ran, what didn't, and the caveats.

## Layout
- `code/` — `run_probe.py` (the harness: action-level branch resampling on SHADE) · `adaptive_probe_DESIGNED_not_run_on_shade.py` (the adaptive engine; core, still to run) · `run_generation.sh` (plain-resampling baseline) · `analyze.py`, `efficiency_sim.py`, `replay_through_harness.py`, `shade_to_inspect.py`.
- `data/` — the one completed task's transcripts (spam_filter_update, 5 benign + 5 rogue), a harness demo trajectory, Inspect traces (`shade_probe.eval`), and the task split.
- `docs/` — STORY, VERIFICATION, RESULTS_pilot (+ BS audit), REVIEW_method_vs_results, RESULTS_SECTION_draft, REPORT_TEMPLATE, REFERENCES, PREREG, PLAN, and `review.ipynb`.

## Reproduce the results (CPU, minutes, no model calls)
Point the analysis at the transcripts in `data/` and run: `python code/analyze.py`, `python code/efficiency_sim.py`,
`python code/replay_through_harness.py`. View traces with an Inspect viewer on `data/shade_probe.eval`.

## Reproduce the runs (needs the model + benchmark)
Serve a black-box model (we used qwen3.6-35b on local vLLM, OpenAI-compatible) and set up SHADE-Arena
(github: jkutaso/SHADE-Arena). **The SHADE-on-aarch64 install recipe is withheld from this public repo per the
sprint's disclosure rule** (do not release novel installation recipes without review); available on request
pending review.

## Status & attribution
Sprint artifact, honest and unfinished. Author: Zachary Gunderson. Pipeline/harness built with Claude Code.
See `docs/REPORT_TEMPLATE.md` for the writeup scaffold.
