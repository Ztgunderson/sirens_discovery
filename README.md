# Sirens for Agents

Resampling-derived, action-level honeypot discovery, piloted on one SHADE-Arena task.

[Report](https://ztgunderson.github.io/sirens_discovery/) | [Apart project](https://apartresearch.com/sprints/projects/discovering-honeypots-from-resampled-behavior-in-evaluation-w9u3) | [Cite](#cite)

Status: parked pilot

## Abstract

A siren is a honeypot for an agent: a tool or file that a well-behaved agent has no reason to touch, so that using it
is a monitoring signal. Because the covert actions cannot be listed in advance, this work tries to discover them.
It resamples a black-box agent at the action level, without reading its chain of thought, and looks for actions that
benign runs never take. The pilot covers one task from SHADE-Arena with one local open-weights model. A simple
detector built from the benign runs flagged both completed sabotage runs and none of the benign runs, but the harm
in this task needs a dedicated tool, so the result is largely structural and the sample is tiny. The adaptive engine
designed to make discovery cheap never finished a full episode on the available hardware.

## Repository layout

```
code/    harness (run_probe.py), adaptive engine (designed, not run on SHADE), analysis scripts, make_fig1.py
data/    transcripts for the one completed task (5 benign, 5 rogue), split.json, demo trajectory, shade_probe.eval
docs/    report source (index.html), figure 1, review.ipynb
SETUP_WITHHELD_NOTE.txt   note that part of the setup is intentionally withheld
```

## Reproduce

Most scripts in `code/` (`analyze.py`, `efficiency_sim.py`, `replay_through_harness.py`) and `docs/review.ipynb`
expect a `shade/` checkout that is not in this repo, so they cannot run from a clean clone. The model runs also
need that setup, which is intentionally withheld (see `SETUP_WITHHELD_NOTE.txt`).

What does run from a clean clone, from the repository root, is the figure script, which reads only `data/transcripts/`:

```
python code/make_fig1.py    # rewrites docs/fig1_efficiency.png
```

## Limitations

- One task, few runs per arm, two positives: no useful intervals, and no held-out evaluation was run.
- The detector works largely by construction, since the side task requires a dedicated tool.
- The adaptive engine is released but was never validated on SHADE.

## Cite

This continues the Apart project linked above.

```bibtex
@misc{gunderson2026sirens,
  author = {Gunderson, Zachary},
  title  = {Sirens for Agents: Resampling-Derived Honeypot Design for Action-Level Detection},
  year   = {2026},
  url    = {https://ztgunderson.github.io/sirens_discovery/}
}
```

Contact: https://ztgunderson.github.io/
