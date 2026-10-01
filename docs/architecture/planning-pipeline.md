# Planning pipeline

Planning is intentionally staged so a fast numerical score cannot override policy or
correctness.

1. **Resolve configuration.** Determine profile, owner limits and explicit user intent.
2. **Resolve legal candidates.** Apply enabled state, allowed/required/excluded policy,
   trust constraints and execution mode.
3. **Check capability evidence.** Reject devices/backends that cannot execute the
   requested model semantics.
4. **Build model manifest.** Represent legal partition units, persistent state,
   workspace and communication boundaries.
5. **Apply memory admission.** Use physical-pool budgets, not summed device VRAM labels.
6. **Enumerate/construct bounded candidate plans.** Search is finite and controlled by
   candidate/deadline budgets.
7. **Route transfers using actual topology semantics.** Logical links do not imply
   independent physical bandwidth.
8. **Score according to objective.** Capacity, interactive latency and throughput are
   distinct objectives.
9. **Validate through the backend adapter.** The adapter either accepts the exact plan
   or returns a structured rejection.
10. **Persist an immutable plan record** before execution.

When the search budget is exhausted, the result is `SEARCH_INCOMPLETE`, not proof that
no feasible plan exists.
