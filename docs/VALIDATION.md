# VALIDATION — introspection-verbalization

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: The repository is a competent infrastructure sketch, but it contains no experiment implementation and its current design is too circular, underspecified, and underpowered to support the claimed introspection result.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Stages and domain pipeline are implemented; X1–X13 spine fixes are present (exact pins, MPS fallback, chat templates, hooks/patching, layer validation, MDE/TOST, n_items=512). Pilot is intentionally synthetic-first. Matches Codex v2 PASS_WITH_NOTES.

### Remaining
- Pilot collect defaults to synthetic activations unless measured weights are explicitly requested.
- Several HF model revisions (including pilot Qwen) remain pinned to `main` rather than immutable commit SHAs (X9 partial).

## Reconciliation
Codex v1 SERIOUS_PROBLEMS → Codex v2 PASS_WITH_NOTES after stage implementation and X1–X13. Grok concurs: no blocking engineering gaps; residual synthetic-default and revision=`main` notes only.
