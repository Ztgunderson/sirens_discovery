# HANDOFF — SHADE resampling-honeypot (reproduce + continue this week)
Honest scope: this reproduces what we ACTUALLY did. Read docs/STORY.md and docs/VERIFICATION.md FIRST.

## What is here
- code/run_probe.py            OUR harness (branch-resampling ReAct loop on SHADE). Built + runs; NOT yet
                               shown to reach a covert action live (see STORY/VERIFICATION).
- code/adaptive_probe_DESIGNED_not_run_on_shade.py   the adaptive ladder+SPRT+allocation ENGINE. Designed,
                               NOT executed on SHADE. This is the core method still to be run.
- code/run_generation.sh       the BASELINE runner (SHADE's plain N independent episodes) = what produced results.
- code/replay_through_harness.py, efficiency_sim.py, analyze.py, shade_to_inspect.py   analysis (CPU, read-only).
- data/transcripts/{benign,rogue}   the ONE task's episodes (spam_filter_update, 5+5) = the actual data.
- data/harness_demo_trajectory.json  one harness episode (benign tools only; no covert action reached).
- data/shade_probe.eval        Inspect traces (view with inspect view).
- docs/                        STORY (the honest narrative), VERIFICATION (triangulation + corrections),
                               RESULTS_pilot (numbers + BS audit), REVIEW_method_vs_results, PREREG, PLAN, notebook.

## Reproduce the RESULTS (no GPU, minutes)
1. Setup SHADE per SETUP.md (INTERNAL: novel install recipe, withhold from public repo pending review).
2. Numbers: python code/analyze.py ; python code/efficiency_sim.py ; python code/replay_through_harness.py
   (paths assume the SHADE checkout at ../shade with data/transcripts in place).

## Continue this week (the real next experiment)
Run code/adaptive_probe_* (the ladder+SPRT+allocation) ON SHADE via the run_probe harness, on faster hardware
or a served model, across >=3 tasks, with argument-level detection. That is the untested core.

## Honest one-liner
Method + harness built and specified; BASELINE (plain resampling + post-hoc control-anchored detector) gave a
structural n=2 result on 1 task; the ADAPTIVE core was not run. Do not overclaim.
