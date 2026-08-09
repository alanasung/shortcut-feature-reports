# VALIDATION — introspection-verbalization

## Codex (p3)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants model-measured behavioral GT, live activation-patch report regeneration, and fail-closed LoRA — standards partly beyond the local measurable pilot scope; proxy behavioral fields and honesty-claim gating are residual notes.
- Detail: `orchestration/out/validate/introspection-verbalization.json`

## Grok (p3 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Measured path is real for a local M4 pilot (smoke-only synthetic, fail-closed collect, pinned chat-template activations, train-split probes, peft LoRA, honesty refuses without live regen, FEATURE holdout). Codex SERIOUS_PROBLEMS treated as frontier-purity gaps, not empty stages.
- Detail: `orchestration/out/grok/validate/introspection-verbalization.p3.md`

## KEY_FIXES (p3)
| Fix | Status |
|---|---|
| Real activation collect + chat templates | OK (`activations.py`, `model_runtime.py`) |
| `force_synthetic` smoke-only; pilot measured default | OK (`pipeline._force_synthetic`, smoke/pilot yaml) |
| Fail-closed measured collect (no silent synthetic) | OK (`RuntimeError` when weights missing) |
| Probes on behavioral GT; train-split fit | OK (`probes.py`, `data.behavioral_label`) |
| LoRA path with peft / labeled fallback | OK (`train.py`) |
| Honesty claim refuses without live regen | OK (`causal.py` `requires_live_regen`) |
| Held-out FEATURE generalization | OK (`FEATURES`, holdout_is_feature) |
| Pinned Qwen/gpt2/pythia/llama revisions | OK (configs + registry SHAs) |
| Hub-free domain tests (monkeypatch) | OK (`test_domain_p3_measured.py`) |

## Remaining (compute / scale — not empty stages)
- Live patch→regenerate-report path not yet wired; honesty claim correctly gated off.
- Behavioral GT fields are generator-defined (independent of probes) but not yet from live with/without-hint model runs.
- Evaluate baselines still use `synthetic_verbalize` for shortcut controls.
- Codex power/purity concerns accepted as residual notes for the local pilot.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable core. Codex SERIOUS_PROBLEMS remains on frontier behavioral measurement and live causal regen — recorded as residual scale notes, not missing stages. Domain tests pass (63).

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 rigor pass (live paired GT + honesty + untrained baseline)

| Fix | Status |
|---|---|
| Smoke dataset stamps `behavioral_gt_source=planted` | OK (`data.py`) |
| Measured collect: paired hint vs no-hint → `hint_flips_answer`, `behavioral_gt_source=live_paired` | OK (`data.apply_live_paired_behavioral_gt`, `pipeline.stage_collect`) |
| Live honesty path: no silent `except: pass`; `live_regen_failed` status | OK (`causal.py`) |
| Evaluate threads runtime when not synthetic; honesty gated | OK (`pipeline.stage_evaluate`) |
| Measured untrained baseline via real `model_verbalize` (never plant ~0.52) | OK (`train.py`) |
| Synthetic baselines stamped `is_synthetic=True` and excluded from headline deltas when measured | OK (`pipeline.stage_evaluate`) |
| Hub-free P6 domain tests | OK (`tests/test_domain_p6_live_gt.py`) |

Residual (scale, not empty stages): full live GT over entire n_items on large models; powered multi-seed honesty regen.

