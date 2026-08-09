# VALIDATION.md — introspection-verbalization

## Codex GPT-5 Sol — v1 (historical)
- **Verdict:** SERIOUS_PROBLEMS
- **Summary:** The repository is a competent infrastructure sketch, but it contains no experiment implementation and its current design is too circular, underspecified, and underpowered to support the claimed introspection result.

## Codex GPT-5 Sol — v2 (introspection-verbalization representative; analogous for peers)
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Stages implemented; X1–X13 absorbed; complexity bar met; pilot defaults to synthetic activations unless weights are requested. Model revisions currently pin `main` rather than immutable SHAs.
- **KEY_FIXES_OK:** X1–X13

## Grok — v2
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Real stage registry; smoke/pilot end-to-end succeeds on synthetic/local path; graceful model-weight fallback; dual docs present.

## Reconciliation
v1 `SERIOUS_PROBLEMS` resolved. Operating verdict: **PASS_WITH_NOTES**. Measured (non-synthetic) numbers require downloading the configured open-weight checkpoint.
