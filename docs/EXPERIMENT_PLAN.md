# Experiment plan

Stage-by-stage design. Each stage is registered in `src/introspect/stages.py`
and appears in `python -m introspect stages`.

## Stages

| stage | responsibility |
|---|---|
| `data` | cued-bias dataset construction, hint injection, train/test splits |
| `activations` | forward-hook capture of residual stream at chosen layers |
| `probes` | linear probe fitting, cross-validation, calibration |
| `verbalize` | prompt templates + parsing of the model's self-report |
| `train` | LoRA fine-tuning loop on probe-derived labels |
| `causal` | ablation and activation patching interventions |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m introspect run -c configs/pilot.yaml --stage causal
```

## Controls and their purpose

- The probe may track the input hint token rather than an internal decision state, which would make 'introspection' just input echoing. The no-hint arm, the input-only baseline, and patching separate these.
- LoRA can teach a surface template ('I used the hint') keyed on lexical cues. Whole-feature holdout plus the metadata-only baseline is the guard.
- A linearly decodable direction need not be causally operative. The ablation must be shown to change model BEHAVIOR, not just the probe readout, before the direction is treated as a real internal feature.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
