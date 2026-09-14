# The weekend, honestly (2026-09-13)

## What we WANTED to test
Resource-efficient resampling at scale: use ADAPTIVE resampling (branch/pause-and-fork at decision points +
a control-anchored novelty ladder + SPRT stopping + adaptive allocation) to DISCOVER, at the ACTION level
(no chain-of-thought), which covert affordances a black-box model reaches for under incentive, cheaply, then
distill that into a PURE honeypot a third party can deploy with a stated false-positive rate. Motivation: the
incident showed CoT was untraceable (30-90%) and there was no trajectory monitoring.

## What we actually did (a chain of forced pivots)
1. Authored "coordination mirror" testbench -> found it BIASED: we authored the target; de-seeding the planted
   protocol made the "covert coordination" largely vanish, i.e. it was mostly instruction-following.
2. ExploitBench (the benchmark we wanted) -> UNRUNNABLE here: amd64-only V8 images, ~70GB/bug, aarch64 + 26GB
   free. Blocked on hardware.
3. Pivoted to SHADE-Arena (third-party, mock tools, runs locally, GROUND-TRUTH labels). Stood it up on local
   vLLM (qwen3.6-35b).
4. Overnight run used SHADE'S OWN runner = PLAIN resampling, NOT our adaptive method. Only 1 of 5 tasks
   finished (spam_filter_update, 5 benign + 5 rogue): episodes are 200-346 turns, ~50 min each -> the resource
   wall hit hard.
5. Post-hoc analysis of that 1 task: a control-anchored action detector (benign never uses the autodelete
   tool) flags both completed-harm positives, 0/5 benign; SHADE's own monitor MISSED one (verdict 0 vs 95).
6. Built the adaptive HARNESS and ran it live on SHADE: it integrates (drives model through SHADE, records
   branches/rungs/scores), BUT a steered full episode TIMED OUT at 3h (reached turn 65, looping on
   preview_rule), never reached the covert action; steering was INERT. No live positive.

## Method we WANTED vs method we TESTED
- Wanted: adaptive, resource-efficient resampling that discovers + detects cheaply and beats brute force.
- Tested: (a) PLAIN resampling + post-hoc control-anchored detection on 1 task; (b) the adaptive harness RUNS
  but was NOT shown to work or help - it never completed an episode, steering didn't help, it costs more.
  => the headline method is DESIGNED and INTEGRATED, not validated.

## What we REALLY saw
1. The resource wall is the central finding: long-horizon frontier-designed benchmarks run ~1 task/night on
   local hardware, and branch-resampling is even slower (couldn't finish one episode in 3h).
2. A no-CoT, control-anchored action detector CAN give zero false positives WHEN the harm needs a distinct
   affordance benign work never uses - but that is STRUCTURAL to the task, n=2, not a general result.
3. SHADE's LLM monitor is inconsistent (missed a real harmful completion) -> a niche for an action tripwire.
4. Authored testbenches mislead: our first "covert coordination" was mostly following a planted protocol.
5. The adaptive method's promised wins (efficiency, steering to rare events) were NOT demonstrated; evidence
   leans against them on this hardware.

## Limitations
1 task; n=2 positives (no powered AUC); detection is structural (harm has a dedicated tool) and won't transfer
where harm reuses benign tools; transcripts log names only (harness records args but never reached a covert
action to capture them); adaptive method never completed a live episode; single model, mock-tool sandbox, one
benchmark; much standup done by subagents (core numbers re-verified by hand).

## What we REALLY want (goal)
A deployable, black-box, action-level (no-CoT) honeypot/tripwire a third party can run to flag rogue agent
behavior at a stated FPR, with the bait discovered efficiently by resampling.

## Next steps (what more resources/time add)
- Faster hardware or a served/faster model (the resource wall is binding).
- Argument-level detection (harness already logs args) on tasks where harm reuses benign tools.
- More tasks + positives for a powered leave-one-task-out ROC.
- Actually run the adaptive allocation/SPRT/subset-sim at scale to TEST the efficiency claim (untested).
- A discovery test on a bench not built to hand harm a dedicated tool.
