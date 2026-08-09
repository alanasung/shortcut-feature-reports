# Training Models to Verbalize Internal Activations

**Target project.** Introspection Training for Verbalization Activations
**Research areas.** Chain of thought; Mechanistic interpretability; Scalable oversight

## Summary

Use linear-probe readouts of a model's own residual stream as cheap ground truth, and train the model to verbalize them honestly.

## Hypothesis

A model can be trained to report the state of an internal feature it was not previously verbalizing, and that ability generalizes to held-out features rather than collapsing into a surface heuristic that mimics the probe without reading the underlying activation.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. Without any training, how accurately can a model already verbalize the readout of a probe trained on its own activations? This is the untrained in-context baseline the prior work asked applicants to propose.
2. Does LoRA fine-tuning on probe-derived labels improve honest verbalization, and does the gain transfer to features held out of training?
3. Is a trained report causally downstream of the activation, or merely correlated with the input? Ablating or patching the feature should move the verbal report if the model is genuinely introspecting.

## Method

1. Build a cued-bias dataset in the style of Turpin et al. 2305.04388: multiple-choice items presented with and without a planted hint.
2. Collect residual-stream activations at several layers for both arms.
3. Fit linear probes for 'the hint influenced this answer', validated against the behavioral ground truth that the hint actually flipped it.
4. Measure untrained in-context verbalization accuracy against probe labels.
5. LoRA fine-tune the model to emit a calibrated verbal report of the probe readout, holding out a subset of features entirely.
6. Run ablation and activation patching as the causal check.

## Measurements

- verbalization accuracy vs probe label
- expected calibration error of stated confidence
- held-out feature generalization gap
- causal sensitivity: change in report under feature ablation

## Threats to validity

- The probe may track the input hint token rather than an internal decision state, which would make 'introspection' just input echoing. The no-hint arm, the input-only baseline, and patching separate these.
- LoRA can teach a surface template ('I used the hint') keyed on lexical cues. Whole-feature holdout plus the metadata-only baseline is the guard.
- A linearly decodable direction need not be causally operative. The ablation must be shown to change model BEHAVIOR, not just the probe readout, before the direction is treated as a real internal feature.

## Report baselines

- input-only: a model given the prompt but not the activation, to show the report is not recoverable from the input text alone
- metadata-only: a model given surface metadata (hint present or absent) without the activation, to catch label-shortcut learning
- shuffled-activation: correct format, wrong activation, as the null

## Feature set and what 'held out' means

Feature-level generalization requires more than one feature, so the design uses a family of at least five distinct internal features with separate probes: hint reliance, answer certainty, format compliance, a planted topical concept, and sycophantic agreement. Held-out means an entire FEATURE is withheld from training, not merely held-out examples of the same feature. Holding out examples would only test in-distribution generalization and would not support the introspection claim.

## Untrained in-context protocol

the motivating application question asks for the untrained in-context experiment, so it is specified rather than gestured at: k-shot demonstrations at k in {0, 2, 8}, a fixed report schema with an explicit confidence field, calibrated against the probe label, with train/test splits partitioned by feature and by prompt template so no template seen in demonstrations appears at test time.

## Literature engagement

docs/RELATED_WORK.md contains a written critique of arXiv:2511.08579, which the prior work lists as optional but recommended, focused on the difference between interpreting an injected decontextualized activation and reporting one's own naturally occurring internal state.

## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
