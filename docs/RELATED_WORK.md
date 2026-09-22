# Related work

**Review date:** 10 August 2026
**Scope:** primary papers and official repositories for the agent benchmarks
closest to OperateBench's construct.

This document exists to make the OperateBench claim narrow and checkable. Every
component of the design has prior art, and most of it has better-established
prior art than ours. What follows is written to concede that first and
differentiate second.

> **Source verification.** Every URL in [Sources](#sources) was re-checked on 10
> August 2026 by a bounded HTTP GET; all 20 resolved with HTTP 200. Each of the
> 12 arXiv identifiers was additionally checked against the arXiv API, and every
> returned title matches the work cited here. No source was inaccessible, and
> none has been dropped. Re-run both checks before any later publication: a
> citation that no longer resolves is recorded as inaccessible, never silently
> removed.

## The claim, stated narrowly

> OperateBench evaluates whether an agent can own a business operation as it
> evolves across time, actors, authoritative systems, waits, human checkpoints
> and external events — not merely complete a task or a sequence of tasks.

That sentence describes the **target design**. What this preview executes is a
proper subset of it, and the two are kept apart throughout this document: see
[Executable in this preview](#executable-in-this-preview) and
[Target design, not yet measured](#target-design-not-yet-measured).

This is a **synthesis and differentiation** claim, scoped to the benchmarks
reviewed here, to the date of the review, and to the properties this build
actually executes. It is not a claim to be first, broadest, most realistic or
industry-standard, and it is not yet validated. It becomes a category claim only
if matched controls show that the longitudinal environment adds signal beyond a
decomposed or static version of the same case (see
[METHODOLOGY.md](METHODOLOGY.md#10-matched-controls)) — and those controls have
not been run.

## Prior art we do not claim

Each of the following is established work, and OperateBench claims none of it as
novel:

- dynamic simulated user–agent conversation with policy and tools — τ-bench [1];
- a shared environment both user and agent change through tools — τ²-bench [2];
- long-horizon, economically valuable professional workflows in real OS
  sandboxes with verifiable deliverables — Agents' Last Exam [12][13][14];
- realistic CRM/business tasks with confidentiality evaluation —
  CRMArena-Pro [3];
- stateful tools with intermediate milestone evaluation — ToolSandbox [6][17];
- compositional enterprise knowledge-work tasks — WorkArena++ [4][19];
- a self-contained simulated workplace — TheAgentCompany [5];
- state-based evaluation with collateral-damage checks — AppWorld [7][18];
- sequential operations-management decisions under uncertainty — AIM-Bench [10];
- "longitudinal" professional task streams and self-evolution —
  FinEvo-Bench [8];
- multi-month workflow memory and continuation — ContextWeave [9];
- controlled failure injection, cascade radius and recovery measurement —
  OrchestraBench [11].

Two words in particular are already taken and are used here only with an
immediate qualifier: **longitudinal** (FinEvo-Bench and ContextWeave use it for
learning across tasks; we mean *within one operation*) and **long-horizon**
(ALE's meaning — a large professional deliverable — is the better-established
one).

## Comparison

The "difference" column compares against **what this preview executes**, not
against the target design. Where a row says OperateBench does something, that
something is in the shipped build and asserted by the offline suite. The two
target-design properties — provider cost in the result vector, and matched
Boundary/Lifecycle controls — are excluded from every comparison below, because
neither exists yet and neither can differentiate anything. They are listed under
[Target design, not yet measured](#target-design-not-yet-measured).

| Benchmark | Core idea | Genuinely similar | Difference from OperateBench |
|---|---|---|---|
| **τ-bench** [1] | Dynamic user–agent conversations under domain policy with API tools; evaluation compares final database state against an annotated goal, with `pass^k` for consistency. | Service domains, policy following, mutable database, user simulation, deterministic outcome evaluation, reliability across trials. | The unit is one conversational session. The world does not primarily advance through days, timers, exogenous events and repeated invocations; WAIT, reopen/finality and typed human load are not the construct. |
| **τ²-bench** [2][16] | Dual-control environment where user and agent both act through tools; Telecom modelled as a Dec-POMDP with compositional tasks and a tool-constrained user simulator. | Multiple active actors, shared mutable state, coordination, programmatic task generation, fine-grained failure attribution. Closest interaction model to ours. | Still an interactive episode around one support task. Persistent case ownership across simulated operational time, deadlines, approvals, post-commit effects and reopenings is not the object of measurement. |
| **Agents' Last Exam (ALE)** [12][13][14] | 1K+ long-horizon, economically valuable professional task instances across 55 subfields and 13 industry clusters, sourced from expert-completed projects and executed in real OS sandboxes with verifiable outcomes. | End-to-end professional workflows, persistent sandbox state, hidden references, deterministic graders where possible, reproducible provisioning, broad harness/tool surfaces. The strongest positioning neighbour. | A workflow instance is one concrete input/output pair: `load()` creates a deterministic start, the agent runs one action loop, `evaluate()` scores the deliverable or resulting state. It does not primarily model an independently evolving multi-actor case with semantic event time, first-class WAIT, authority and human checkpoints, reopen/finality or Human Load. |
| **CRMArena-Pro** [3][20] | 19 expert-validated sales, service and CPQ tasks over B2B/B2C scenarios, with persona-guided multi-turn interaction and confidentiality tests. | Enterprise workflows, multi-turn users, permissions and confidentiality, realistic structured business state. Closest business setting. | Measures professional tasks inside a CRM. There is no operation episode with an independent event queue and clock, WAIT obligations, typed human checkpoints, repeated case cycles or replay-final terminality. |
| **ToolSandbox** [6][17] | Stateful conversational tool-use benchmark with implicit state dependencies, on-policy user simulation and milestone-DAG evaluation over arbitrary trajectories. | Stateful tools, interactive user, intermediate milestones, multiple valid paths, evaluation beyond final text. Closest runtime primitive set. | Built around tool-use capability rather than ownership of a semantically modelled business operation over days, with external actors and systems, authority and human load. |
| **WorkArena++** [4][19] | 682 realistic enterprise knowledge-work tasks performed through web applications; compositional planning, reasoning and retrieval, with ground-truth trace generation. | Real workplace workflows, compositional tasks, rich environment interaction, executable traces. | Each item is a task to finish. The environment is not an independently evolving case with event-time semantics, WAIT and case finality. |
| **TheAgentCompany** [5] | Self-contained simulated software company where agents browse, code, run programs and communicate with simulated coworkers. | Workplace realism, multiple apps and communication surfaces, long-horizon actions, consequential side effects. | Broad digital-work automation rather than a typed operation lifecycle. Completion stays task-centric; authority, timed external events, checkpoints and replay-final state are not the core construct. |
| **AppWorld** [7][18] | Controllable world of nine apps and 457 APIs with 750 interactive coding tasks; state-based unit tests accept different solutions and check collateral damage. | Rich mutable world, many APIs, programmatic state evaluation, multiple valid trajectories, unintended-change detection. | A complex digital task remains the unit. No first-class event-driven operational time, WAIT, human-checkpoint ledger or multi-invocation case ownership. |
| **AIM-Bench** [10] | Repeated inventory-replenishment experiments probing long-term planning and decision biases under uncertainty. | A genuine operations-management setting, sequential decisions, uncertain evolving state, long-term consequences. Closest operations-research neighbour. | Focuses on replenishment policy and bias across experiments, not one multi-actor case with communications, evidence, approvals, authoritative systems, reopen and human intervention. |
| **FinEvo-Bench** [8] | A longitudinal sequence of 120 related financial tasks measuring whether self-evolving agents use earlier experience to improve later work, with compliance rubrics and paired non-evolving controls. | Professional workflows, longitudinal ordering, compliance, evolving memory, paired controls. | Longitudinal means learning **across distinct tasks**. OperateBench means ownership **within one evolving operation**; self-improvement is not part of the initial construct. A positioning collision with a different unit. |
| **ContextWeave** [9] | Privacy-preserved multi-month office-work streams reconstructed into 1,005 executable tasks, testing whether recalled experience improves downstream work and resists misleading recall. | Multi-month context, workflow continuation, executable environments, persistence and memory, preference alignment. The strongest workflow-continuity neighbour. | The unit is again a stream of downstream tasks. One business operation does not persist through independent events, obligations, authority changes and terminal guards. |
| **OrchestraBench** [11] | Controlled failure injection into templated enterprise multi-agent workflows; measures cascade radius, per-failure recovery, decomposition and routing quality. | Recovery, trusted state, failure attribution, enterprise framing, reproducible seeds, causal mechanism probes. A close neighbour for recovery methodology. | The object of evaluation is the orchestrator or pipeline, not an agent owning a business operation. Most useful to us as prior art to build on rather than to differentiate from. |

## What is left after the concessions

The target design is a combination of nine properties held together as one
benchmark contract. **Seven of them are executable in this preview; two are
design, not build.** They are separated below because a comparison table is only
worth reading if the things being compared exist, and because the two missing
properties are precisely the ones that would carry the most weight.

### Executable in this preview

Every property here is exercised by the shipped operation and asserted by the
offline suite. These are the only properties the [Comparison](#comparison) above
differentiates on.

1. **Operation as the execution unit.** One `OperationEpisode` survives many
   agent invocations.
2. **Environment independence.** Simulated time, actors, timers and
   authoritative systems evolve even when the agent does nothing.
3. **WAIT as a scored action.** Correct waiting binds a wake condition and a
   deadline; waiting forever and premature follow-up are different failures.
4. **An authority and provenance ladder.** Participant claims, extracted
   evidence, human approvals and authoritative-system facts are not
   interchangeable.
5. **Human checkpoints as typed state.** Required approval, justified exception,
   unnecessary escalation, missed escalation and transfer are measured
   separately.
6. **Commit and side-effect separation.** Proposal, authority validation,
   committed business change and post-commit external failure are distinct
   records.
7. **Loops, reopen and finality.** A case can revisit work, close provisionally,
   reopen, and later become replay-final; late events cannot silently resurrect
   it.

### Target design, not yet measured

Both of these are named in the Architecture RFC and neither ships. Nothing in
this preview should be read as evidence about them.

8. **An operation-level result vector that includes provider cost.** The vector
   ships and travels with every run — outcome, critical invariants, temporal
   correctness, authority, action validity, Human Load, recovery, obligations
   and deterministic replay. **Provider cost is not part of the published
   benchmark result vector, and no public benchmark artifact reports a
   provider-backed capability result.** Private diagnostics are
   not benchmark capability or public evidence. The publication manifest includes
   no private diagnostic aggregates, raw evidence, paths or provider prose. Cost per successful operation is a target-design field, not a
   current published result.
9. **Boundary/Lifecycle matched controls.** *Designed, not run.* The static or
   decomposed, Boundary-only and time-removed controls described in
   [METHODOLOGY.md](METHODOLOGY.md#10-matched-controls) have not been executed
   against any scenario, so whether longitudinal ownership adds information
   beyond isolated consequential decisions is an open question here, not a
   property this build has.

### What may and may not be claimed from this

The differentiation claim in this document is about the seven executable
properties, scoped to the reviewed set and to the review date. **No claim is
made that the current preview combines all nine**, that it is the first or
broadest benchmark to combine any of them, or that any benchmark in the reviewed
set fails to combine them. Cost measurement and matched controls are the two
things that would make the combination worth arguing about, and they are exactly
the two that have not been built — which is an argument for building them, not a
result.

## Known positioning risks

- **"τ-bench plus a clock."** τ-bench already supplies dynamic service
  conversation, tools, policy and database-state evaluation [1]. Timestamps on a
  static transcript would not answer this. The fixture must contain causally
  load-bearing exogenous events, correct WAIT, deadlines, authority changes,
  post-commit failure and reopen/finality — and the controls must show it.
- **Category overlap with ALE.** ALE targets sustained, economically valuable
  professional work at a scale we cannot match on breadth, expert count or
  general workplace realism [12][13]. Our category is narrower by choice: live
  operation ownership under an independently changing world.
- **Enterprise realism is not our moat.** CRMArena-Pro, WorkArena++ and
  TheAgentCompany have broader task and environment coverage [3][4][5]. We lead
  with causal operational semantics, not breadth.
- **Recovery needs causal probes, not failure counts.** OrchestraBench already
  measures failure modes, cascade radius and recovery [11]. Recovery evaluation
  must tie recovery to canonical business state, idempotency, obligations, actor
  authority and post-commit effects.

## What we adopt from this work

- From ALE: a hard separation of specification, agent, environment and grading;
  gate-then-score evaluation; deterministic checks over LLM judges; published
  provenance and correction policy; separate code and data licences [12].
- From τ²/τ³: actor simulators constrained to authorised actions and visible
  state; separate spec/runtime/evaluator versions with an explicit comparability
  statement; acceptance of equivalent valid paths [2][16].
- From ToolSandbox: per-transition state snapshots, an obligation/milestone DAG,
  and explicit scenario capability tags [6][17].
- From AppWorld: evaluator assertions over both required and forbidden state
  deltas, so collateral damage fails [7][18].
- From WorkArena++: extracting reviewed domain-neutral primitives before
  composing new scenarios from them [4][19].
- From CRMArena-Pro: actor-visible fields and disclosure rules declared per
  operation, with targeted negatives for cross-actor leakage [3][20].
- From OrchestraBench: single named faults at exact causal boundaries, detection
  latency, blind retry as a negative control, paired seeds [11].
- From ContextWeave and FinEvo-Bench: matched controls for stale or misleading
  derived context, and interleaving to reduce order effects [8][9].

What we deliberately do not adopt: breadth before construct validity; production
records as public fixtures; heavy GUI/VM infrastructure in v0; final-state-only
success; LLM judges for core truth; and a leaderboard before incremental
longitudinal signal is demonstrated.

## Sources

1. τ-bench: Tool-Agent-User Interaction — <https://arxiv.org/abs/2406.12045>
2. τ²-bench: Dual-Control Conversational Agents — <https://arxiv.org/abs/2506.07982>
3. CRMArena-Pro — <https://arxiv.org/abs/2505.18878>
4. WorkArena++ — <https://arxiv.org/abs/2407.05291>
5. TheAgentCompany — <https://arxiv.org/abs/2412.14161>
6. ToolSandbox — <https://arxiv.org/abs/2408.04682>
7. AppWorld — <https://arxiv.org/abs/2407.18901>
8. FinEvo-Bench — <https://arxiv.org/abs/2608.06144>
9. ContextWeave — <https://arxiv.org/abs/2608.04830>
10. AIM-Bench — <https://arxiv.org/abs/2508.11416>
11. OrchestraBench — <https://arxiv.org/abs/2608.05263>
12. Agents' Last Exam (paper) — <https://arxiv.org/abs/2606.05405>
13. Agents' Last Exam (website) — <https://agents-last-exam.org>
14. Agents' Last Exam (repository) — <https://github.com/rdi-berkeley/agents-last-exam>
15. τ-bench repository — <https://github.com/sierra-research/tau-bench>
16. τ²-bench repository — <https://github.com/sierra-research/tau2-bench>
17. ToolSandbox repository — <https://github.com/apple/ToolSandbox>
18. AppWorld repository — <https://github.com/StonyBrookNLP/appworld>
19. WorkArena repository — <https://github.com/ServiceNow/WorkArena>
20. CRMArena repository — <https://github.com/SalesforceAIResearch/CRMArena>

Corrections to this page — a mischaracterised benchmark, a missed neighbour, a
claim we have not earned — are welcome as issues or pull requests. See
[../CONTRIBUTING.md](../CONTRIBUTING.md).
