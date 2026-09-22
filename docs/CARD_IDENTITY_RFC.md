# Card identity RFC

**Status:** Phase A normative identity foundation; Artifact 9 implementation pending
**RFC version:** 1
**Date:** 21 August 2026
**Identity consumer:** [IDENTITY_REPLAY_RFC.md](IDENTITY_REPLAY_RFC.md)

This RFC freezes the minimum identity-bearing projections for operation core,
Semantic Scenario, and concrete Variant before Artifact 9. It does not define the
later human authoring workflow or a broad publication card. Phase E may add
non-identity public/controlled fields but MUST NOT change these identity semantics.

An additive Card v2 chain is now frozen beside v1: Operation v2, Semantic Scenario
v2, and Variant v2 must reference only the same v2 generation. Card v1 resources,
bytes, schema literals, validation behavior, and the public `CARD_SCHEMA_VERSION = 1`
alias remain frozen; there is no migration or implicit upgrade. V2 adds only the
`NULLABLE_STRING` field shape (explicit JSON null or a string satisfying every
published string constraint) and the `EXACTLY_ONE` quantified predicate (exactly
one matching object value, with only that value bound). This stage is contract-only:
it adds no runtime projector, evidence, replay, compatibility Decision, or Artifact
9 behavior. Runtime identity adapter v2 resources are deliberately deferred until
the Maintenance declaration source leaves exist and can be frozen without invention.

The additive v2 wire set is
[Operation Card v2](schemas/operation-card-v2.schema.json),
[Semantic Scenario Card v2](schemas/semantic-scenario-card-v2.schema.json),
[Variant Card v2](schemas/variant-card-v2.schema.json),
[shared definitions](schemas/card-defs-v2.schema.json),
[registry](schemas/card-registry-v2.json), and the exact
[field census](schemas/identity-field-census-v2.registry.json) with its
[schema](schemas/identity-field-census-v2.schema.json). Their relative references
resolve offline to the sibling v2 definitions. The v1 contract below remains
normative and unchanged.

The text/path loader is deliberately versioned. Legacy `load_card_json` remains
v1-only and accepts historically valid JSON with arbitrary legal surrounding
whitespace; it performs no pre-parse raw-byte cap. `load_card_json_v2` is v2-only
and refuses raw UTF-8 input larger than 8 MiB before parsing. Neither loader
accepts the other generation or performs migration.

### Normative schema set and precedence

The following checked-in Draft 2020-12 documents are the exact wire contract:

- [Operation Card v1 schema](schemas/operation-card-v1.schema.json), `$id`
  `https://operatebench.org/schemas/operation-card-v1.schema.json`;
- [Semantic Scenario Card v1 schema](schemas/semantic-scenario-card-v1.schema.json),
  `$id` `https://operatebench.org/schemas/semantic-scenario-card-v1.schema.json`;
- [Variant Card v1 schema](schemas/variant-card-v1.schema.json), `$id`
  `https://operatebench.org/schemas/variant-card-v1.schema.json`;
- [shared Card v1 definitions](schemas/card-defs-v1.schema.json), `$id`
  `https://operatebench.org/schemas/card-defs-v1.schema.json`; and
- [Card v1 registry](schemas/card-registry-v1.json), including the exact schema
  subset keyword registry, set/ordered array registry, predicate operators, field
  shape tags, and operation-policy registrations.

Every relative `$ref` is exactly `card-defs-v1.schema.json#/$defs/...` and MUST be
resolved to that sibling file by matching its pinned `$id`; network retrieval and
replacement by a same-named remote document are forbidden. The JSON Schemas and
registry are normative. Pseudotypes in this RFC are explanatory projections only;
if prose admits a shape, keyword, operator, optional field, timestamp, or policy
member absent from those artifacts, it is refused. Cross-record invariants that
JSON Schema cannot express (ordering, reference resolution, acyclicity, digest
equality, and operation-contract conformance) remain normative in this RFC.

## 1. Rules common to all cards

### 1.1 Strict envelope

Every card is a JSON object with exactly:

```text
{
  schema: namespaced schema literal,
  schema_version: 1,
  id: namespaced ID,
  content: schema-specific object,
  content_digest_sha256: 64 lowercase hexadecimal characters
}
```

No field is optional. Unknown fields at any depth are refused. JSON values are
restricted to string, integer, boolean, null where explicitly allowed, arrays, and
objects with string keys. Floats, NaN/infinity, duplicate JSON keys, lone
surrogates, non-string keys, and integers outside the signed 64-bit range are
refused. Empty identifiers and uncontrolled prose are not identity fields.

IDs match:

```text
^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*(?:@[0-9]+\.[0-9]+\.[0-9]+)?$
```

Schema literals are:

- `operatebench.operation_card.v1`;
- `operatebench.semantic_scenario_card.v1`; and
- `operatebench.variant_card.v1`.

An ID is a human reference, not content identity. Two equal IDs with different
content digests conflict and fail closed.

All card timestamps use exactly `YYYY-MM-DDThh:mm:ss.ffffffZ`: UTC `Z` and exactly
six fractional digits. Equivalent offsets, omitted/shorter/longer fractions, leap
seconds, and impossible Gregorian dates are refused. JSON Schema enforces the
lexical form; loaders also validate the calendar date.

### 1.2 Canonicalization and digest

Cards use the canonical JSON rules in the Identity Replay RFC: sorted object keys,
no insignificant whitespace, UTF-8, arrays preserved unless declared set-class,
and strict JSON scalar safety. A set-class array is duplicate-free and sorted by
canonical encoded bytes before hashing. Ordered causal arrays retain authored
order and MUST NOT be sorted.

The digest is:

```text
SHA256(canonical_json({
  "domain": schema,
  "schema_version": 1,
  "id": id,
  "content": content
}))
```

`content_digest_sha256` excludes itself. Envelope extensions are prohibited, so
there is no unsigned metadata beside identity. Public/controlled authoring
metadata added later lives in separate schemas and references this digest.

### 1.3 Reference form

Cross-card references are exact objects:

```text
CardRefV1 = {
  schema: exact target schema,
  id: exact target id,
  content_digest_sha256: exact target digest
}
```

A reference never imports target content into the referring card's owned digest.
Loaders resolve it and require exact schema/ID/digest equality. Missing, ambiguous,
or incompatible targets fail closed.

## 2. Operation core card

`operatebench.operation_card.v1` owns stable operation semantics shared by every
scenario and variant. `content` has exactly:

```text
{
  operation_type: namespaced string,
  operation_version: semver string,
  state_contract: {
    schema_id: namespaced string,
    initial_state_schema: strict JSON Schema subset object,
    canonical_state_projection: sorted unique list<JSON pointer>,
    replay_final_pointer: JSON pointer,
    terminal_pointer: JSON pointer
  },
  authority_contract: {
    roles: sorted unique list<RoleV1>,
    event_authority: sorted unique list<EventAuthorityV1>,
    action_authority: sorted unique list<ActionAuthorityV1>
  },
  policy_contract: strict typed object of substantive policy fields,
  event_contracts: sorted unique list<EventContractV1>,
  action_contracts: sorted unique list<ActionContractV1>,
  retrieval_contract: {
    catalogue: sorted unique list<RetrievalContractV1>,
    record_identity_algorithm: namespaced string,
    evidence_requirements: sorted unique list<EvidenceRequirementV1>
  },
  transition_contract: {
    reducer_ids: sorted unique list<namespaced string>,
    obligation_types: sorted unique list<namespaced string>,
    terminal_kinds: sorted unique list<namespaced string>,
    common_predicate_inputs: sorted unique list<JSON pointer>,
    declared_max_domain_generated_audit_events: non-negative signed-64-bit integer
  }
}
```

Typed records are closed:

```text
RoleV1 = {role_id, authorities: sorted unique list<string>}
EventAuthorityV1 = {event_type, required_authority}
ActionAuthorityV1 = {action_type, required_authority}
EventContractV1 = {event_type, required: closed field-shape map,
                   optional: closed field-shape map}
ActionContractV1 = {action_type, required: closed field-shape map,
                    optional: closed field-shape map,
                    evidence_refs: "required" | "optional" | "forbidden"}
RetrievalContractV1 = {tool_id, source, authority, schema_id,
                       arguments: closed field-shape map}
EvidenceRequirementV1 = {outcome_type, selector_id,
                         required_tools: sorted unique list<tool_id>}
```

The strict JSON Schema subset itself is identity-bearing. It is the closed
recursive union at `card-defs-v1.schema.json#/$defs/jsonSchemaSubset`; its exact
allowed and forbidden keywords are duplicated for machine audit in the Card v1
registry. `$ref`, composition, annotations, conditionals, arbitrary extension
keywords, and permissive object schemas are refused. Every object in the subset
has `additionalProperties: false`. `FieldShapeV1` and `FieldShapeMapV1` are the
closed recursive tagged unions in the same definitions file; no prose type name,
Python type, or unregistered tag is accepted.

Every operation-authored `pattern` in that JSON Schema subset or in a field shape
uses the registry-pinned safe-regex contract. Pattern syntax is limited to
printable ASCII characters U+0020 through U+007E. Consequently each syntax
character occupies exactly one UTF-8 byte, so the limits of 256 characters and
256 UTF-8 bytes are equivalent; matched inputs are limited to 4096 characters.
The published authored-schema syntax is boundary anchors plus concatenations of
literals, escaped literals and `dDsSwW` categories, and character classes/ranges.
Literal class atoms are complete; the portable `0-9`, `A-Z`, `a-z`, and full
printable-ASCII ranges are authorable. Other raw ranges and escaped range
endpoints are not. Repeat bounds use canonical decimal spelling (no leading zero
except `0`). Fixed repeats use `{n}`. Variable bounded repeats cover one-digit
ordered bounds, bounds of differing decimal widths, zero minima, and the 4096
ceiling. Each atom may have at most one
simple or bounded quantifier (with an upper bound of 4096), and a pattern may
contain at most one variable-width quantifier; exact bounded repeats do not
branch. The runtime parser also accepts non-canonical leading-zero bounds and a
set of additional valid bounded-repeat and class forms, but those compatibility
forms are deliberately not authorable in
the published schema. Thus schema acceptance is a sound subset of runtime
acceptance. Every grouping construct, alternation, backreference, lookaround,
inline flag, conditional, atomic group, advanced escape, wildcard, adjacent
quantified atom, repeated quantifier, second variable-width quantifier, and
malformed escape/class is refused before matching.
Packaged Card schemas are immutable
normative resources and remain on
the trusted schema-validation path; this authored-pattern subset does not
reinterpret or reject their pinned patterns.

`policy_contract` is not generic. V1 registers exactly
`operatebench.policy.lettings.maintenance.v1` for operation type
`lettings.maintenance.synthetic`; its twelve required fields, positive-integer bounds, and
currency/name shapes are frozen in the Operation Card schema and registry from
`spec.py::_POLICY_FIELDS`, `PolicySpec`, and `_parse_policy`. A new operation type
requires a new closed schema branch and registry entry before use. The transition bound
`declared_max_domain_generated_audit_events` is operation-owned, required, and
non-negative. Under
[`operatebench.domain-generated-audit-budget.v1`](DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md)
it is exactly the authored maximum number of distinct logical operation-authored
audit-event intents that may be successfully committed through the future Core
budget API for one operation instance. Card owns only this aggregate scalar: source
semantics, taxonomy, counters, reservations, findings, and evidence remain outside
the Card. The value is deliberately not a policy-duration field and MUST NOT be
inferred from a run. Operation Card v2 needs no schema change for this meaning.

The operation card excludes scenario IDs, authored events, absolute starts/times,
surface fixtures, expected terminal for a case, model/scaffold/provider settings,
evaluator predicates/results, and build provenance. Adding an unrelated scenario
therefore cannot change operation core identity.

### Baseline ownership map

The migration source is `domains/lettings/maintenance/spec.py` for event/action
payload schemas, actors/authority, and policy; `operation.py`/`state.py` for
reducers, canonical state, obligations and terminal semantics; and
`retrieval.py`/`core/read_contract.py` for retrieval/evidence contracts. Phase B
MUST project owned semantic fields explicitly; file/import closure is forbidden.

## 3. Semantic Scenario card

`operatebench.semantic_scenario_card.v1` owns causal meaning and authored hazards
shared by all concrete surfaces/variants of one scenario. `content` has exactly:

```text
{
  operation_core: CardRefV1,
  construct_label: namespaced string,
  common_estimand_id: namespaced string,
  decision_points: ordered unique list<SemanticDecisionPointV1>,
  causal_graph: {
    nodes: sorted unique list<CausalNodeV1>,
    edges: sorted unique list<CausalEdgeV1>
  },
  authored_hazards: sorted unique list<AuthoredHazardV1>,
  common_outcome_obligations: sorted unique list<CommonObligationV1>,
  allowed_variant_dimensions: sorted unique list<VariantDimensionV1>
}
```

```text
SemanticDecisionPointV1 = {
  point_id, order,
  reach_predicate: closed predicate AST,
  required_prior_point_ids: sorted unique list<point_id>,
  common_obligation_id,
  admissible_effect_classes: sorted unique list<string>
}
CausalNodeV1 = {node_id, semantic_type}
CausalEdgeV1 = {cause_node_id, effect_node_id, relation}
AuthoredHazardV1 = {hazard_id, predicate: closed predicate AST,
                    expected_common_finding_code}
CommonObligationV1 = {obligation_id, predicate_id, terminal_required: boolean}
VariantDimensionV1 = {dimension_id, value_type: closed enum,
                      changes_semantics: false}
```

The predicate AST is exactly
`card-defs-v1.schema.json#/$defs/predicateAst`: the tags are `all`, `any`, `not`,
`scalar`, and `quantified`; scalar operators are `EQ`, `NE`, `LT`, `LTE`, `GT`,
`GTE`, `IN`, `NOT_IN`, and `MATCHES`; quantifiers are `ALL`, `ANY`, and `NONE`;
and value-expression tags are `pointer`, `binding`, and `literal`. Each branch's
required fields and `additionalProperties: false` are schema-frozen. It may refer only to operation-core
canonical pointers and bindings. Unknown operators, evaluator function names,
Python callables, or prose selectors are refused.

`decision_points` order is semantic and retained. Every `point_id`, node, hazard,
and obligation is unique. Causal edges must resolve, be acyclic, and connect the
frozen points/hazards. Allowed variant dimensions must explicitly state that they
do not change semantics; any proposed dimension that changes event type,
authority, policy threshold, causal edge, obligation, or common predicate belongs
in a new Semantic Scenario.

The Semantic Scenario excludes concrete timestamps, payload values, actor IDs,
fixture text IDs, dispatch-failure selections, event IDs, provider/model settings,
arm-specific time transforms, and evaluator output.

### Baseline ownership map

The baseline migration sources are `semantic_arms_v1.py::SemanticScenario` and
`SemanticDecisionPoint`, plus scenario semantic IDs and declared expectations in
`spec.py`. Current `operation_spec_digest_sha256` inside `SemanticScenario` becomes
the exact `operation_core` card reference. Compiler-generated transform
declarations are not scenario-owned; they move to arm protocol identity.

## 4. Variant card

`operatebench.variant_card.v1` owns one concrete world/surface realization of one
Semantic Scenario. `content` has exactly:

```text
{
  semantic_scenario: CardRefV1,
  variant_label: namespaced string,
  starts_at: canonical UTC timestamp,
  expected_terminal: operation-core terminal enum,
  human_checkpoint_budget: non-negative integer,
  required_checkpoint_types: sorted unique list<string>,
  dispatch_failures: sorted unique list<fixture_id>,
  expected_event_rejections: ordered unique list<ExpectedRejectionV1>,
  actors: sorted unique list<VariantActorV1>,
  hidden_initial_values: closed object conforming to operation state schema,
  message_fixtures: sorted unique list<MessageFixtureBindingV1>,
  events: ordered unique list<VariantEventV1>
}
```

```text
ExpectedRejectionV1 = {event_id, code}
VariantActorV1 = {actor_id, role_id}
MessageFixtureBindingV1 = {fixture_id, surface_content_digest_sha256}
VariantEventV1 = {
  event_id, event_type, actor_id, authored_sequence,
  at: canonical UTC timestamp | null,
  trigger: VariantTriggerV1 | null,
  delay_minutes: non-negative integer,
  triggers_agent: boolean,
  payload: closed object conforming to referenced event contract
}
VariantTriggerV1 = {
  kind: "action" | "event",
  type_name,
  correlation: closed map of operation-declared binding fields
}
```

Exactly one of `at` and `trigger` is non-null. Absolute events are ordered by
`(at, authored_sequence, event_id)`; the stored `events` array MUST already be in
canonical authored order. Conditional events are ordered by authored sequence and
ID within their causal owner. Event IDs and authored sequences are unique where
the operation contract requires them. Actor roles must resolve to the operation
card, event authority must hold, event types/payloads must resolve, triggers must
resolve, expected rejection IDs must exist, and fixture IDs must resolve.

`surface_content_digest_sha256` identities content held in a separate public or
controlled fixture store. Surface bytes are not imported into semantic scenario
identity. A surface change moves the Variant digest through its binding, never the
operation or semantic scenario digest.

The Variant excludes runtime output, accepted effects, trial/run IDs, model
responses, evaluator grades, provider evidence, pricing, scaffold, arm protocol,
and build provenance.

### Baseline ownership map

The baseline source is `spec.py::ScenarioSpec` and the authored fixture YAML;
`MaintenanceOperation.build_plan` is a consumer, not owner. Current scenario data
mixed into the whole-spec digest is projected here. The Variant references the
Semantic Scenario digest rather than causing an unrelated variant or scenario to
move historical identity.

## 5. Ownership and dependency graph

```text
OperationCardV1
  <-ref- SemanticScenarioCardV1
           <-ref- VariantCardV1
```

An arrow is an exact compatibility reference, not ownership transfer. Operation
core owns generic domain semantics. Semantic Scenario owns causal case meaning.
Variant owns concrete world and surface bindings. Every source field maps to one
and only one of those owners or to a different Identity Manifest component.

Phase B's machine-readable field census MUST reject:

- a source field projected into two cards;
- a semantic source field projected into none;
- a card field sourced from runtime/evaluator output;
- a reference whose target cannot be resolved exactly; and
- a change to one card that unexpectedly moves an upstream card.

The census contract, initial exhaustive Artifact 9 root registry, and expansion
rules are respectively
[identity-field-census-v1.schema.json](schemas/identity-field-census-v1.schema.json)
and
[identity-field-census-v1.registry.json](schemas/identity-field-census-v1.registry.json).
B1 MUST package these artifacts with all four card schema/registry documents and
MUST replace each `map_values`/`array_items` pattern with the complete expanded
leaf census for the concrete Artifact 9 schema before Artifact 9 identity code can
depend on it. Zero-owner, multiple-owner, absent-consumer, stale source-path, and
unexpanded-pattern cases fail CI.

Consumers are: Identity Manifest and Artifact 9 writer/reader (all three), runtime
compiler (all three), evaluator compatibility checks (operation/scenario), arm
compiler/verifier (scenario/variant), provider run plan (exact references), and
analysis/reporting (references only).

## 6. Public and controlled projections

These v1 cards are identity projections and contain no free-form descriptions,
reviewer names, controlled execution evidence, or secrets. A later public Operation/Scenario
Card and a controlled supplement MAY reference the exact v1 digest and add
non-identity metadata. They MUST state whether each added field is public or
controlled and MUST NOT be accepted as an identity card by an Artifact 9 reader.

Result Card is deferred from this identity foundation. When implemented under
`operatebench.result_card.v1`, it must separately represent execution status,
scientific verdict, original/regrade lineage, and limitations, and reference—not
redefine—the three identities here.

## 7. Migration and lifecycle

1. Freeze baseline source fixtures and derive candidate cards with an explicit
   source-path-to-owner report.
2. Independently recompute each projection from source semantics and compare.
3. Verify that adding an unrelated scenario/variant leaves historical Operation
   and Semantic Scenario identities unchanged as applicable.
4. Ship readers for exact v1 only; unknown schema versions fail closed.
5. Artifact 8 remains unchanged and continues to carry its historical whole-spec
   digest. It is never rewritten to contain these cards.
6. Artifact 9 contains exact card references and the Identity Manifest. No fallback
   derives v1 identities from Artifact 8 at read time.

Compatible metadata additions use a separate referencing schema, not optional
fields in v1. Any change to an identity-bearing field moves its content digest.
Any change to canonicalization, field ownership, included fields, or meaning is an
incompatible new named schema and Artifact contract; it cannot be called v1 with a
minor version bump.

## 8. Tests-to-be and phase boundary

Before Artifact 9, tests MUST cover exact unknown-field refusal at every depth,
duplicate JSON keys, scalar safety, ID grammar, set ordering/duplicates, ordered
causal arrays, digest self-exclusion, domain separation, cross-reference mismatch,
DAG cycles, predicate-AST unknown tags/operators, authority/payload mismatch,
invalid event trigger shape, and unresolved fixture binding.

Mutation tests MUST show: operation semantics move all dependent bundles but not
unrelated card content; scenario semantics move that scenario and variants only;
variant event/surface changes move that variant only; evaluator/scaffold/provider
changes move no card; unrelated scenario addition leaves historical identities
fixed. A source/owner census must fail on every unowned or multiply owned field.

This RFC is normative Phase A text. Executable card/census test results are the
required implementation evidence. It does not itself make current fixtures Artifact 9,
authorize a provider run, validate the construct, or support ranking/generalization.
