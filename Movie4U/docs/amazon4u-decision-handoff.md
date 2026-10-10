# Pending Amazon4U decision record — temporary handoff

This note preserves an agreed Amazon4U decision while active editing instructions
permit changes only inside Movie4U. It does **not** change Movie4U's evaluation
protocol or authorize Amazon4U training. Once scope instructions are resolved,
transfer this decision to Amazon4U's training-semantics documentation and relevant
configuration, then retire this temporary note.

## Agreed: profile validation before designing a workaround

- Start with full-ranking validation every epoch on Musical_Instruments.
- Before comparative experiments, profile end-to-end validation time, including
  embedding inference, candidate/history filtering, ranking and metric calculation.
- Measure peak memory and validation time relative to training-epoch time.
- Keep the per-epoch schedule if practical; do not design a workaround in advance.
- Reassess feasibility when scaling to Toys_and_Games and Electronics. Feasibility
  on Musical_Instruments does not establish feasibility on larger categories.
- Preserve the agreed full-ranking candidate universe; do not silently substitute
  sampled candidates or evaluation users to reduce cost.
- If frequency must change, agree and document the amendment before comparative
  experiments. Early-stopping patience is measured in validation checks, not epochs.

**Rationale:** use resource measurements rather than speculative optimization,
while preserving the pre-specified evaluation semantics.

**Status:** user-approved approach; profiling not performed. The final validation
schedule awaits those measurements. The proposed numerical early-stopping patience
and other training hyperparameters are not approved by this decision.

## Remaining action

Resolve editing scope, move this record to its authoritative Amazon4U documents
and configuration, perform profiling once the training/evaluation implementation
and other prerequisites are ready, and document measurements and schedule choice.
No files outside Movie4U were changed to record this handoff.
