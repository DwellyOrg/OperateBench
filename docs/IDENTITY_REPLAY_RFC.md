# Identity and replay RFC

**Status:** normative architecture contract; B2 Identity Manifest Slices 1–5 construction, validation, census, requirement evaluator, and identity diff implemented; final compatibility Decisions and domain adapters pending. B3.3 historical runtime-evidence and identity-adapter contracts and the additive audit-integrity OutcomeV3/ExecutionRecordV2/ExecutedRuntimeEvidenceV2 resources are frozen as schema/resources only; all runtime authority, projectors, binding, evidence construction, replay, compatibility, and capability work remains unimplemented
**RFC version:** 3
**Date:** 25 August 2026
**Card foundation:** [CARD_IDENTITY_RFC.md](CARD_IDENTITY_RFC.md)
**Measurement boundary:** [MEASUREMENT_RFC_V2.md](MEASUREMENT_RFC_V2.md)

This RFC selects one architecture. Alternatives are not interchangeable at
runtime. Normative types below are documentation-level executable interfaces:
Phase B may choose Python spelling, but MUST preserve field, ownership, dependency,
immutability, refusal, and process-boundary semantics.

The additive immutable Card v2 Operation → Semantic Scenario → Variant contract is
now checked in while every Card v1 resource and behavior remains frozen. V2 permits
only the new `NULLABLE_STRING` shape and `EXACTLY_ONE` quantifier described by the
Card RFC, refuses mixed-generation references, and provides no migration path. This
is a contract-only stage: no runtime or identity projector, evidence constructor,
replay path, compatibility Decision, or Artifact 9 implementation consumes it.
Runtime identity adapter v2 schemas/registries are blocked on real Maintenance
source declarations and are deferred rather than synthesized from absent leaves.
The legacy `load_card_json` API remains v1-only and preserves whitespace-padded
historical JSON without a pre-parse raw-byte cap; bounded v2 ingestion uses the
separate v2-only `load_card_json_v2` API with an 8 MiB pre-parse UTF-8 limit. This
API split does not add migration, runtime consumption, or projectors.

## 1. Baseline and required separation

The current baseline is Engine `0.9.0`, Maintenance `0.6.0`, Artifact `8`, model
protocol v4, request mapping v7, and execution ledger `3`. Current
`EpisodeRun` in `src/operatebench/runner.py` combines `EpisodeOutcome`, decision
tape, execution record, and `OperationEvaluation`; `artifact.py` re-executes and
compares a complete artifact including evaluation. `engine_version` names runtime
and evaluator together.

That baseline is evidence to preserve, not this contract's implementation. Artifact
9 MUST separate execution from evaluation, exact runtime replay from historical
evaluator matching, and compatibility from comparability.

## 2. Identity Manifest v1

`operatebench.identity_manifest.v1` is a strict canonical JSON document. It has
exactly:

```text
{
  schema: "operatebench.identity_manifest.v1",
  schema_version: 1,
  components: closed map<ComponentName, ComponentIdentityV1>,
  composites: closed map<BundleName, CompositeIdentityV1>,
  compatibility_contract: CompatibilityContractV1,
  manifest_digest_sha256: Digest
}

ComponentIdentityV1 = {
  schema: "operatebench.component_identity.v1",
  schema_version: 1,
  domain: exact component domain from identity-component-registry-v1.json,
  content_digest_sha256: Digest,
  owned_fields: sorted unique list<FieldPath>,
  owned_projection: map<FieldPath, JSONValue>,
  excludes: [],
  requires: direct dependencies in exact registry order
}

DependencyRequirementV1 = {
  component: ComponentName,
  compatible: ExactDigestRequirementV1
            | ClosedSchemaVersionRangeRequirementV1
            | NamedPredicateRequirementV1
}

ExactDigestRequirementV1 = {
  kind: "exact_digest", content_digest_sha256: Digest
}
ClosedSchemaVersionRangeRequirementV1 = {
  kind: "closed_schema_version_range",
  minimum_schema_version: positive integer,
  maximum_schema_version: positive integer
}
NamedPredicateRequirementV1 = {
  kind: "named_predicate", predicate: dotted namespaced string,
  predicate_version: positive integer
}

CompositeIdentityV1 = {
  schema: "operatebench.composite_identity.v1",
  schema_version: 1,
  domain: exact composite domain from identity-component-registry-v1.json,
  members: ordered unique list<ComponentName | BundleName>,
  bundle_digest_sha256: Digest
}
```

The exact closed document shapes, component and composite names, domains, edges,
member order, and requirement variants are normative in
[the Identity Manifest v1 JSON Schema](schemas/identity-manifest-v1.schema.json)
and
[the component registry](schemas/identity-component-registry-v1.json). The built-in
compatibility contract is exactly `operatebench.compatibility_contract.v1`, schema
version `1`; its requirement kinds are `exact_digest`,
`closed_schema_version_range`, and `named_predicate`, each at version `1`; its
decision predicates are `can_runtime_replay`, `can_evaluator_regrade`,
`can_analysis_recompute`, `can_compare`, and `can_pool`, each at version `1`; and
its named-requirement-predicate registry is empty. The complete frozen object is
normative in
[the compatibility registry](schemas/identity-compatibility-v1.registry.json).

Unknown fields, duplicate names/paths, unsorted set-class arrays, invalid domains,
missing required components, nullable required identities, self-reference, cycles,
unowned fields, and multiply owned fields MUST be refused.

### 2.1 Canonical identity rule

Every digest uses SHA-256 over canonical UTF-8 JSON: every object key, including
keys in the component and composite maps, is sorted lexicographically on the wire;
arrays preserve their contract-defined order; booleans/strings/null and semantic
integers only; no fractional or non-finite numbers, lone surrogates, or insignificant
whitespace. Semantic integers are bounded to the interoperable JSON range
`-(2^53-1)..+(2^53-1)`. Runtime accepts an exact Python `int` (but not `bool`) or a
finite, mathematically integral Python `float` in that range and normalizes it to an
exact integer before canonicalization. Thus nested `1.0` and `1` have identical
canonical bytes and digests, while `1.5`, NaN/infinity, and out-of-range integers are
refused. This narrow normalization is specific to Identity Manifest v1; the B1 Card
contract's no-float rule is unchanged.

Identity Manifest v1 additionally caps an owned projection at 4,096 fields, any
array or object at 4,096 direct items/properties, and an individual string at exactly
1 MiB of UTF-8 bytes at runtime. A complete normalized component identity — including
its owned projection, owned fields, exclusions, and requirement metadata — is capped
by `MAX_COMPONENT_IDENTITY_BYTES = 336 * 1024` canonical UTF-8 bytes and
`MAX_COMPONENT_IDENTITY_NODES = 5600` strict-JSON nodes. These component limits are
derived for the mandatory complete eleven-component manifest, rather than only its
eight-member runtime subset: eleven individually valid components leave deterministic
wrapper headroom beneath the unchanged global manifest envelope of 4 MiB and 65,536
nodes. Component construction applies the complete-identity envelope before returning,
which also guarantees that the runtime and experiment bundles fit their global resource
envelopes.

JSON Schema `maxLength` counts code points rather than UTF-8 bytes, so the recursive
schema uses the conservative authoring limit 262,144 code points for strings and
object property names. Every schema-accepted string therefore satisfies the runtime
byte cap; runtime may accept longer ASCII strings as a compatibility superset. The
schema uses an ECMA-262-compatible negative lookahead absolute-end assertion rather
than `$`, so newline-suffixed digests, names, predicates, and pointers cannot cross a
schema/runtime full-match boundary. JSON Schema cannot express the aggregate
component or composite byte/node envelopes; runtime semantic validation is normative
for those limits. Existing JSON nesting and field-path limits are 64 and 1,024
characters respectively. Public strict-JSON inputs use exact built-in `dict` and
`list` containers: tuples and container subclasses are refused. Immutable tuples and
mapping proxies exist only inside validated library records. Packaged normative
resources are read incrementally in requested chunks no larger than 64 KiB and refused
above 4 MiB before hashing or parsing. A resource stream gets at most 128 `read()` calls;
each call must return exact built-in `bytes`, with `b''` as EOF, and may not return more
bytes than requested. Cleanup cannot mask a primary read or protocol failure. With no
primary failure, an ordinary close failure crosses the same stable resource-error
boundary, while exception groups and control-flow `BaseException` instances propagate
unchanged. These are semantic refusal limits, not implementation guidance. The exact
wrappers are:

```text
component = {
  "domain": COMPONENT_DOMAIN,
  "schema": "operatebench.component_identity.v1",
  "content": OWNED_PROJECTION
}
composite = {
  "domain": COMPOSITE_DOMAIN,
  "schema": "operatebench.composite_identity.v1",
  "content": {
    "members": [{"name": MEMBER_NAME, "identity": COMPLETE_MEMBER_IDENTITY}, ...]
  }
}
manifest = {
  "domain": "operatebench.digest.identity_manifest.v1",
  "schema": "operatebench.identity_manifest.v1",
  "content": MANIFEST_WITHOUT_MANIFEST_DIGEST_SHA256
}
```

The digest field itself, signatures, and mutable storage metadata are excluded
unless this RFC explicitly includes them. Domain strings MUST NOT be reused across
component, bundle, correction, report, or grade identities.

A component projection is a flat map from concrete, non-root RFC 6901 JSON
Pointer `FieldPath` to the detached value at that path. B2 inherits B1's escaped
non-empty-token subset: every reference token has at least one character, `~0`
and `~1` are the only escapes, and therefore `/` and `/a/` are refused. It is not a reconstructed
nested document. `owned_fields` is exactly the sorted key set of that map. A
component content digest covers only this flat projection. A
`requires` reference is compatibility metadata, not imported owned content. A
composite digest covers each ordered member name and its complete identity object,
including requirement metadata and owned projection, plus the composite's own
domain/schema. The manifest digest excludes only `manifest_digest_sha256` from
its content. Runtime mutation therefore moves runtime content and dependent
bundles but not evaluator content; evaluator mutation does the converse.

### 2.2 Required components and DAG

The required component names and their direct `requires` edges are exactly:

```text
card_schema_content          -> (none)
build_provenance_content     -> (none)
operation_core_content       -> card_schema_content
semantic_scenario_content    -> operation_core_content, card_schema_content
variant_content              -> semantic_scenario_content, card_schema_content
runtime_contract_content     -> operation_core_content, card_schema_content
scaffold_content             -> runtime_contract_content
arm_protocol_content         -> runtime_contract_content,
                                semantic_scenario_content, variant_content
provider_run_plan_content    -> runtime_contract_content, scaffold_content,
                                build_provenance_content
evaluator_bundle_content     -> operation_core_content, card_schema_content,
                                runtime_contract_content
analysis_contract_content    -> evaluator_bundle_content
```

Arrows mean “requires,” never “owns.” The two required composites are exactly:

```text
runtime_bundle = ordered(
  operation_core_content, semantic_scenario_content, variant_content,
  runtime_contract_content, scaffold_content, arm_protocol_content,
  provider_run_plan_content, build_provenance_content
)

experiment_bundle = ordered(
  runtime_bundle, evaluator_bundle_content, analysis_contract_content
)
```

There are no implicit edges from a composite back to one of its members.
`evaluator_bundle_content` requires compatible runtime evidence and operation/card
schemas but MUST NOT be a member required to perform runtime replay.
`analysis_contract_content` consumes grades but cannot influence runtime or
evaluator identity. The graph MUST be topologically validated before any digest
is accepted, and the listed topological order MUST be reproducible without file-
or import-closure inference.

### 2.3 Recompute and mutation interface

```text
recompute_component(name, owned_projection, requires) -> ComponentIdentityV1
recompute_bundle(name, resolved_member_identities) -> CompositeIdentityV1
identity_to_json(identity, *, resolved_member_identities=None) -> JSON object
validate_identity_manifest(manifest, field_census, owned_projections)
  -> ValidatedIdentityManifest
identity_diff(old, new) -> tuple<OwnedMutation, ...>
```

`validate_identity_manifest` is the implemented B2 Slice 3 trust boundary.
`owned_projections` supplies the
independently derived flat maps that the validator compares with each serialized
component projection. It MUST NOT derive ownership by importing Artifact, runtime,
evaluator, provider, or card implementations.

All three validator roots and every caller JSON object/array are exact built-in
`dict`/`list` values. The projection root has exactly the eleven component keys in
the frozen registry and each value is an exact plain flat object; empty projections
are legal. Validation proceeds in frozen registry topological order. It first joins
the census, independent projection keys, and serialized `owned_fields` and
`owned_projection`; recomputes each complete component from the independent value
and serialized ordered `requires`; checks each exact digest or closed version range
against the recomputed target (and rejects `named_predicate` while its registry is
empty); recomputes both bundles solely from those leaves; then recomputes the
manifest self-digest with the exact frozen compatibility object. This deterministic
requirement satisfaction emits no compatibility `Decision` and invokes no
predicate.

Success returns a frozen slotted `ValidatedIdentityManifest` containing schema and
version, registry-ordered mapping proxies of freshly recomputed component and
composite records, a recursively frozen detached compatibility contract, and the
fresh manifest digest. Census metadata and caller objects are not retained. This
slice defines neither a JSON projection for that result nor validator idempotence.

Slice 5 implements `identity_diff` over exact `ValidatedIdentityManifest` inputs.
Both forgeable records are structurally revalidated: exact schema/version and
registry key order, globally unique projection ownership, component recomputation
from projection plus ordered direct requirements, composite recomputation from
leaves, frozen compatibility equality, and fresh manifest self-digest. This is an
internal-consistency check, not a substitute for retained census completeness.

`OWNED_MUTATION_CONTRACT_VERSION` is the exact integer `1`. `OwnedMutation` is a
frozen, slotted dataclass with these fields, in order:

```text
mutation_contract_version: int
owner: str
changed_field_paths: tuple[str, ...]
changed_requirement_components: tuple[str, ...]
old_content_digest_sha256: str
new_content_digest_sha256: str
changed_composites: tuple[str, ...]
```

There is one row per changed owner, in component registry order. Concrete path
additions, removals, and strict normalized-JSON value changes are lexical and
unique. Equality is recursively type-sensitive (`bool` differs from `int`, null
differs from absence), ignores object insertion order, preserves array order, and
invokes no caller hooks. A path may be added or removed under one owner, but moving
it between owners or duplicate ownership fails closed. Requirement metadata is not
an owned pseudo-path: changed dependency names follow the owner's direct registry
`requires` order. Requirement-only rows have equal old/new content digests.

`changed_composites` is the owner's reverse membership closure, filtered by actual
fresh old/new composite-record difference and emitted in composite registry order.
Card-schema changes have no composite assignment; runtime and build-provenance
leaves can name `runtime_bundle` then `experiment_bundle`; evaluator and analysis
leaves can name only `experiment_bundle`. Requirement edges are never traversed for
this field. Exact-digest target changes instead create separate requirement-only
rows in direct dependents where their metadata changes.

Identical valid manifests return exact `()`. A stale component, composite, or
manifest digest, changed compatibility contract, reordered/malformed requirement,
projection-identical content-digest difference, or reassigned owner is malformed;
none creates a synthetic row. Manifest-digest and composite-only consequences do
not create rows. The diff invokes no requirement evaluator, compatibility Decision,
replay, comparability, provider, or filesystem logic, has no JSON projection, and
retains no input projection or requirement values. Domain projection adapters and
Artifact 9 remain later work.

For `runtime_bundle`, `resolved_member_identities` is the exact eight-component
leaf closure listed above. For `experiment_bundle`, it is those same eight leaves
plus `evaluator_bundle_content` and `analysis_contract_content`; the kernel MUST
recompute `runtime_bundle` internally before hashing the experiment's three direct
ordered members. A caller-supplied composite record or lexically valid bundle
digest is never authority. Every supplied component is recomputed from its included
owned projection and exact metadata before use.

`identity_to_json` is projection, not deserialized-record validation. Components
are self-contained and MUST be recomputed before projection. Composite wire records
deliberately do not carry their leaf identities, so projection requires the exact
eight-leaf runtime resolver or ten-leaf experiment resolver. The kernel recomputes
the bundle from those leaves, compares only exact normalized scalar/tuple record
fields, and projects the freshly recomputed result. Object identity, process-local
registries, caller-supplied digests, copied records, and frozen-dataclass mutation
are never provenance. `validate_identity_manifest` establishes trust from
independently supplied projections. Member identities MUST NOT be added to the
composite wire record.

### 2.4 Component requirement evaluation (B2 Slice 4)

`evaluate_component_requirements(requirements, resolved_components)` is the
implemented B2 Slice 4 primitive. It evaluates only the conjunction of explicit
component requirements against freshly recomputed, self-contained component
identities. Exact content digests and inclusive closed schema-version ranges are
supported. The frozen v1 named-requirement-predicate allowlist is empty, so every
well-shaped named requirement is unsatisfied.

The exact ordered, deduplicated reasons are
`unknown_named_requirement_predicate`, `missing_component`,
`content_digest_mismatch`, and `schema_version_out_of_range`. Unknown requirement
kinds and malformed caller envelopes are validation errors, not denial reasons.
The empty requirement list is only the true empty conjunction (`satisfied=True`);
it does not authorize replay, regrading, analysis recomputation, comparison, or
pooling. Predicate vocabulary is not executable predicate semantics. The five final
compatibility Decisions remain specified-only until their canonical owner inputs
and adapters exist.

The frozen, slotted `RequirementEvaluationV1` contains only
`requirement_contract_version=1`, `satisfied`, and `reasons`. Slice 4 consumes the
unchanged vocabulary-only compatibility registry; no mirrored registry resource or
packaged-resource integrity pin changed; it does not change component, composite,
compatibility-contract, or golden manifest bytes/digests. The implementation, RFCs,
dependency metadata, and contract tests are the affected surfaces; no identity or
version repin is required.

`recompute_manifest_digest` is a hash-only helper: it does not validate
component/composite closure or make a manifest authoritative; full validation is
available through `validate_identity_manifest`. The helper operates over bounded
strict JSON, normalizes semantic integers, and excludes only
`manifest_digest_sha256`.

## 3. Current-code ownership and consumer census

This table is the Phase A source map for B1–B5. “Target owner” is normative for
Artifact 9; “current source/consumer” grounds the migration in the baseline.
Grouped paths are exhaustive identity categories, not permission to leave new
fields uncensused.

| Field/category | Current owner/source | Current consumers | Artifact 9 target owner |
|---|---|---|---|
| distribution version | `src/operatebench/version.py::__version__` | build metadata, release scanner | build provenance |
| combined `engine_version` | `version.py::OPERATEBENCH_VERSION` | runner, artifact writer/reader, docs/scanner | legacy alias only; never a comparison key |
| operation id/type/version, actors/authority, policy, action/event schemas, hidden initial domain semantics | `domains/lettings/maintenance/spec.py::OperationSpec` and `operation.py` | runner, Engine/domain, evaluator, compiler, artifact | operation core card |
| `spec_digest_sha256` (currently whole authored mapping) | `spec.py` | runner, Engine identity, artifact replay, controls, evidence recorder, CLI | migrated into card component digests plus explicit compatibility references; legacy field retained only in Artifact 8 reader |
| semantic scenario id, decision predicates/order, authored hazards | `spec.py::ScenarioSpec`; `semantic_arms_v1.py::SemanticScenario` | compiler, controls, runner/evaluator | Semantic Scenario card |
| concrete starts/event payloads/times/surface fixtures/dispatch failures/expected event rejections | `ScenarioSpec` and fixture YAML | `MaintenanceOperation.build_plan`, Engine, tests | Variant card |
| runtime clock, queue ordering, event dispositions, transition records, outcome provenance | `core/clock.py`, `core/events.py`, `core/engine.py`, `core/ledger.py` | runner, artifact replay, evaluator reads runtime evidence | runtime contract |
| model-visible projection, retrieval/action/outcome contracts, model protocol v4 | `core/protocol.py`, `core/read_contract.py`, `core/retrieval*.py`, `core/outcomes.py`, `agents/model.py` | Engine, model adapters, playback, artifact validation, evaluator | runtime contract (scaffold references it; does not own it) |
| Maintenance reducers/guards and replay-final semantics | `maintenance/operation.py`, `state.py` | Engine, reference/negative agents, evaluator evidence | operation core + runtime bundle requirement; each field itself owned once by operation core |
| evaluator predicates, dimensions, finding vocabulary, terminal-suffix mapper, common outcome version | `maintenance/evaluator.py`, `core/evaluation.py`, current matched-arm evaluator code | runner, artifact writer/replay comparison, reports | evaluator bundle; `CommonOutcomeV1` is a separate evaluator component |
| standardized scaffold/instructions/tool presentation | current model request construction in `agents/model.py` and adapter projection | model adapter, request digest/evidence | scaffold content; requires runtime model protocol |
| arm names, compiler, treatment declarations, point correspondence | `maintenance/semantic_arms_v1.py`, later No-Time modules | matched controls/operator, evaluator mappings | arm protocol/compiler component |
| Artifact schema/writer/reader/replay constants | `artifact.py` (`ARTIFACT_VERSION=8`, supported/reproducible tuples) | CLI, runner, execution bundle, tests | Artifact schema + runtime replay contract |
| execution ledger schema and run ID | `execution_ledger.py` (`EXECUTION_LEDGER_VERSION=3`, historical v2 reader, `INTENDED_ARTIFACT_VERSION=8`) | provider evidence recorder, artifact binding, bundle audit | execution-evidence component; Artifact 9 migration MUST move `INTENDED_ARTIFACT_VERSION` to `9` only when B4 writes 9 |
| artifact-ledger binding and bundle minimum | `artifact.py::PROVIDER_EXECUTION_FIELDS`; `execution_bundle.py`, whose current `version >= ARTIFACT_VERSION` check effectively means 8 | artifact validator, bundle auditor | execution-bundle contract with a literal named minimum `ARTIFACT_VERSION_WITH_LEDGER_BINDING = 7`; MUST NOT use moving writer alias as historical threshold |
| provider request mapping v7 and adapter settings | `agents/openai_responses.py` and provider modules | transport, evidence recorder, artifact request digests | provider/model/run-plan component; request mapping also required by runtime/scaffold compatibility |
| provider/model/profile, request settings, retries, caps, pricing digest/rates | provider adapters, `agents/evidence.py`, pricing/cost modules, operator run plan | transport, execution ledger, artifact binding, bundle audit | provider/model/run-plan component; its closed pricing-policy subprojection owns rates and the run-plan fields bind its digest |
| decision tape and execution record | `agents/playback.py`, `agents/transport.py`, `runner.py::ExecutedEpisode/EpisodeRun` | artifact writer/replay, provider binding | executed runtime evidence |
| execution-ledger rows and totals | `execution_ledger.py` | evidence recorder, bundle auditor | executed runtime evidence sidecar under ledger schema identity |
| analysis endpoint/estimand/error/missingness rules | methodology/operator documents | result reporting | analysis contract |
| source revision, dependency lock, wheel/container identity | Git/build/lock/packaging surfaces | operator and release audit | build provenance |
| `operation_instance_id`, execution run id, timestamps, provider call totals, per-run digests | runtime/operator generated | artifact, ledger, bundle, reports | run evidence, never component content identity |

B1 materializes only the exact Card v1 contract census under
`operatebench.b1_card_contract.v1`. That census is deliberately insufficient for
Identity Manifest validation of an Artifact 9 document and MUST be refused when
presented as one. B4 owns the full Artifact 9 field census: every field, including
nested fields and nulls, MUST then have one owner and an enumerated consumer list.
New Artifact 9 fields fail CI until censused. A source file may implement multiple
components, but one field cannot be owned by multiple components. Neither B2 nor a
projection adapter may widen or reinterpret the B1 Card/Census resource contracts.

### 3.1 Manifest-projection census (B2 only)

B2 uses a separate closed census resource whose exact Draft 2020-12 contract is
[the Identity Manifest Projection Census v1 schema](schemas/identity-manifest-projection-census-v1.schema.json).
Its wire shape is exactly:

```json
{
  "schema": "operatebench.identity_manifest_projection_census.v1",
  "schema_version": 1,
  "manifest_schema": "operatebench.identity_manifest.v1",
  "owners": ["the exact eleven component names in registry order"],
  "entries": [{"field_path": "/concrete/pointer", "owner": "component_name"}]
}
```

`owners` is present in exact registry order even when an owner has no fields.
`entries` has at most 4,096 rows, is ordered by owner registry index and then lexical
`field_path`, and contains globally unique concrete paths. For each owner, its row
paths MUST equal both the independent projection key set and the serialized
component's `owned_fields`/`owned_projection` key sets. Values in the serialized
projection MUST equal normalized independent values exactly; `null` is a present
value, while absence is a missing path. Unknown, duplicate, missing, extra, or
multiply owned paths are refused.

This census proves only uniqueness and completeness over the eleven supplied
manifest component projections. It is explicitly neither the B1 Card census nor
the Artifact 9 census, and proves nothing about full caller-domain or Artifact 9
completeness, source paths, consumers, nullability, or cardinality. Its metadata is
not component content and contributes to no digest. B4's full Artifact 9 census
remains independently required. Supplying the B1 schema or its artifact
discriminator to the manifest validator receives an explicit insufficient-scope
refusal; other malformed or unsupported census shapes receive the ordinary Identity
domain refusal.

## 4. B3.3 canonical executed-runtime evidence contract freeze

This section freezes wire content and future adapter obligations only. The schemas and
resources are executable contract artifacts, not package exports or production runtime
types. Nothing in B3.3 mints runtime authority, validates an execution against a
manifest, constructs trusted evidence, replays an artifact, calls an evaluator, or
issues a capability.

### 4.1 Closed JSON profile and immutable projections

The five public records are `EpisodeOutcomeV2`, `DecisionTapeV1`,
`ExecutionRecordV1`, `ProviderExecutionBindingV1`, and
`ExecutedRuntimeEvidenceV1`. Their exact closed Draft 2020-12 schemas are,
respectively:

- [episode-outcome-v2.schema.json](schemas/episode-outcome-v2.schema.json);
- [decision-tape-v1.schema.json](schemas/decision-tape-v1.schema.json);
- [execution-record-v1.schema.json](schemas/execution-record-v1.schema.json);
- [provider-execution-binding-v1.schema.json](schemas/provider-execution-binding-v1.schema.json); and
- [executed-runtime-evidence-v1.schema.json](schemas/executed-runtime-evidence-v1.schema.json).

Unknown or missing fields are refused at every record boundary. Wire containers are
exact built-in JSON objects and arrays; scalars are null, exact bool, exact int, or
Unicode string. Floats, bool-as-int, bytes, tuples, non-string keys, duplicate wire
keys, lone UTF-16 surrogates, cycles, and custom mapping/sequence values are refused.
Integers are bounded to `-(2^53-1)..+(2^53-1)`. Limits are: 64 container levels;
100,000 aggregate nodes per evidence payload; 4,096 direct members per container;
1,048,576 UTF-8 bytes per string or property name; and 8,388,608 canonical UTF-8
bytes per evidence payload. Provider-binding text retains the ledger's stricter
256-character limit. Runtime semantic limits govern where JSON Schema cannot express
UTF-8 bytes, aggregate nodes, or complete payload bytes.

Every exact lexical field is written `^…(?![\s\S])`. A trailing `$` appears in none
of these schemas. ECMA-262 without `/m` treats `$` as an absolute end assertion,
while Python's `re` — the engine behind the `jsonschema` validator this contract
requires — also matches `$` immediately before one final `\n`, so a `$` ending would
make the accepted language depend on which validator ran. The negative-lookahead
ending is absolute in both engines and is the ending already frozen by
[identity-manifest-v1.schema.json](schemas/identity-manifest-v1.schema.json). A
value carrying a trailing `\n`, `\r`, `\r\n`, U+2028 or U+2029 is therefore outside
every identifier, digest, instant and decimal language in this contract, under
either engine.

A future constructor MUST validate and detach exact dict/list JSON, then recursively
freeze objects into fresh read-only mappings and arrays into fresh tuples. Every wire
projection MUST recursively allocate fresh dict/list containers. No mutable child,
caller alias, object identity, or process-local authority marker enters the canonical
graph.

### 4.2 `EpisodeOutcomeV2`

Its fields are exactly `schema`, `schema_version`, `status`, `terminal_outcome`,
`replay_final`, `started_at`, `ended_at`, `simulated_minutes`, `invocations`,
`final_state`, `final_state_digest_sha256`, `trajectory`,
`trajectory_digest_sha256`, and `events`. Schema/version are exactly
`operatebench.episode_outcome.v2`/`2`.

This is a Core-generic contract. A domain terminal identifier has 1–255 characters
and exactly the grammar `^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*(?![\s\S])` — the same
language and length as the existing Variant Card `expected_terminal` contract,
written with the absolute ending of §4.1 rather than a trailing `$`.
`terminal_outcome` is null or one such domain terminal identifier, excluding the two
Engine halt literals. `status` is either `operation_deadlock`,
`operational_horizon_exhausted`, or a domain terminal identifier. The cross-field
state machine is exact:

1. if `terminal_outcome` is null, status MUST be one of the two Engine halts;
2. if status is a business/domain terminal, it MUST equal non-null
   `terminal_outcome` and `replay_final` MUST be true;
3. either Engine halt MAY retain a non-null domain terminal or null, preserving both
   current Engine halt shapes; and
4. a terminal identifier is never inferred from Maintenance's present two values or
   from an evaluator result.

Instants are exact second-precision `YYYY-MM-DDTHH:MM:SSZ`. The pattern fixes the
shape; the semantic layer additionally requires that the value names a real calendar
instant, so `2031-02-31T09:00:00Z` is refused although it satisfies the pattern.
`ended_at >= started_at`; the elapsed interval is a whole number of minutes; and
`simulated_minutes` equals exactly that number of minutes. Every event `at` is a real
calendar instant under the same rule.
Counts are non-negative safe integers. `final_state` is an object and trajectory/events
are arrays under the bounded profile. State and trajectory digests are unwrapped
SHA-256 over their exact canonical bytes. Events retain exactly the current Artifact-8
fields: `event_id`, `event_type`, `actor_id`, `at`, `sequence`, `payload`,
`triggers_agent`, `caused_by`, `disposition`, and `verdict_code`. The selected runtime
bundle remains authority for domain-specific state and trajectory-row variants.
Engine HMAC provenance is a separate private process-local companion, never wire or
digest content; caller-built/wire V2 is content only.

### 4.3 Tape, execution, and provider binding

`DecisionTapeV1` is exactly schema/version, decisions, and decisions digest. Each
closed decision has invocation index, turn index, observation digest, and the current
closed ACT/WAIT/ASK/ESCALATE/COMPLETE/RETRIEVE/MALFORMED outcome union. Indices are
strictly increasing by `(invocation_index, turn_index)` — an equal or descending pair
is refused — and the digest is recomputed over the decisions array.

`ExecutionRecordV1` adds schema/version to the current exact execution fields and
closed attempt fields. Sources are deterministic, model, or recorded. Deterministic
and recorded sources have null model/protocol/token ceiling, zero transport calls,
and no attempts. Model source has non-null non-empty model/protocol, a positive
output ceiling, and attempt/decision cardinality and invocation/turn equality. Retry
count is exactly zero in this contract. Fault, attempt outcome, decision kind and
exclusion are paired exactly, and each pairing is a semantic check:

- an attempt outcome of `decided` or `classified` carries a null `fault`, and an
  attempt outcome of `failed` carries a non-null `fault`;
- because retry count is zero and model source binds attempts to decisions
  one-to-one and in order, a `failed` attempt has no decision to pair with, so
  canonical evidence refuses any `failed` attempt;
- the paired decision's outcome kind and the attempt outcome agree exactly:
  `MALFORMED` pairs with `classified`, and every other kind pairs with `decided`;
- `excluded` false requires a null `exclusion_code` and an empty `exclusion_detail`,
  and `excluded` true requires a non-null `exclusion_code`.

Canonical evidence refuses excluded execution records.

`ProviderExecutionBindingV1` adds schema/version to the exact fifteen Artifact-8
provider-binding fields. It is non-null only for model source; deterministic and
recorded sources require null. Model/binding identity, call/attempt/decision
cardinality, and exact contiguous `decision_call_index` are cross-record checks.

A provider binding may enter trusted evidence **only** when minted from a fresh,
successful audit of a complete scored execution-ledger-v2 or execution-ledger-v3
bundle. Historical v2 remains readable; current writers emit v3. The audit derives
the ledger version from the verified sidecar rows and requires exact equality with
the Artifact binding before it compares totals, calls, and digest. Every identity,
ledger digest/version, provider/API/model, settings/pricing digest, total,
request row, decision row, and call index is compared with the audit's independent
derivation. A mapping, deserialized candidate, artifact field, copied audit summary, or
internally coherent repin is not authority. The audit companion and ledger path are
private, process-local, non-wire and non-digest. Null is permitted only for deterministic
or recorded source; model requires audited non-null. This proves local
artifact/sidecar agreement, not provider attestation. Artifact 8 bytes, names, version,
fixtures, writer, and reader are never changed or upgraded by this migration.

### 4.4 `ExecutedRuntimeEvidenceV1` and digest

Its exact fields are schema/version, `identity_manifest_digest_sha256`,
`operation_instance_id`, outcome, decision tape, execution record, always-present
nullable provider binding, and `runtime_evidence_digest_sha256`. The complete freshly
validated manifest is a construction prerequisite; canonical evidence stores only its
content digest. A validated manifest is adapter output to cross-check, never input
authority. Outcome invocation count must cover the greatest recorded invocation.
No evaluation or analysis field belongs here.

The runtime evidence digest is SHA-256 over `canonical_json_bytes` of exactly:

```json
{"domain":"operatebench.digest.executed_runtime_evidence.v1","schema":"operatebench.executed_runtime_evidence.v1","content":"E with only runtime_evidence_digest_sha256 omitted"}
```

The displayed string describes the projection: on the wire `content` is the complete
object `E`, not a string. Exactly one field—the top-level self-digest—is omitted. Every
explicit null, nested schema/version, manifest digest, operation instance identifier,
provider binding, and nested section digest remains. Process-local Engine provenance,
provider-audit companion, ledger path, storage metadata, and object identities are not
wire fields and therefore are not digest inputs. Canonical keys are lexically sorted,
arrays preserve order, UTF-8 is unescaped for non-ASCII, separators are compact, and
floats/non-finite values/surrogates are refused. Validation recomputes the lowercase
64-hex value.

The [machine-readable golden vectors](schemas/executed-runtime-evidence-v1.golden.json)
contain independently recomputed provider-null and provider-non-null wrappers, exact
UTF-8 text/hex/byte count/hash, boundary constants, and a closed malformed-vector
census. Their hashes are, respectively,
`309b01dd7e08c61e1314b972467c922403f38ef097555ab7f762c125a06c2704`
(1,379 bytes) and
`970ec85c12c5d9bbf562198a3ab016d66abcec2d4abba628e7a6e161419d393e`
(2,415 bytes). The non-null vector is wire/digest conformance only and has no
provider-audit authority.

Those two byte counts and hashes are load-bearing pins, not derived labels. The
executable suite MUST state them, and the load-bearing semantic content of each
vector, as literals authored independently of the golden resource, so that a golden
file whose evidence, canonical text, hex, byte count and stored hash have all been
coherently rewritten together fails rather than re-freezing itself.

That exact-fixture failure is not generic runtime authentication. A content oracle can
recompute section digests, calendar/order/cardinality invariants, and fields duplicated
across the supplied records, but a coherent caller-resealed terminal, state, trajectory,
provider, ledger, settings, or pricing assertion remains content only when no independent
counterpart exists. Exact goldens pin concrete fixture bytes. Future process-local Engine
authentication and runtime replay own terminal/pre-transition execution truth; a fresh
provider-sidecar audit owns request/decision joins, ledger identity and totals, including
request, observation, action, and content digests whenever both records carry them.

Mutation coverage is generated, not declared. The census names each adversarial
class by code — profile, lexical, outcome, tape, execution and evidence — and every
code has exactly one hand-authored constructor in the executable suite. The suite
asserts that its constructor set and the census agree exactly, executes every
constructor, and requires refusal. Beyond the named classes the suite enumerates the
complete recursive pointer set of both valid vectors and mutates every wire leaf and
every wire container, proving the mutated pointer set equals the enumerated one, so
no field can be added without also being mutated. The census covers: every nested and
top-level field; unknown/missing/duplicate keys; bool/int, float, surrogate,
non-string key, non-list sequence, mapping-subclass forgery, unsafe integer,
recursion and all five bounds; the absolute-ending negatives of §4.1;
terminal/status/replay-final transitions; calendar, ordering and elapsed-minute
coherence; strict decision ordering; source nullability; fault/outcome, attempt/
decision-kind and exclusion pairing; invocation coverage; stale and coherently
refreshed section digests; stale self-digest; provider null/source/model/cardinality/
call-index mismatch; and swapped ledger identities, totals, calls, or pricing/settings
digests. Changing only the self-digest must fail and proves it is the sole omission.
Caller aliasing, fresh wire projections and the absence of any audit-companion,
ledger-path or sidecar field on the wire are proved directly: validation MUST NOT
mutate the caller's graph, an aliased sub-graph MUST canonicalize identically to its
detached copy, and any audit or sidecar field added to a record MUST be refused as an
unknown field.

### 4.4.1 Domain-generated audit budget successor dependency

[`operatebench.domain-generated-audit-budget.v1`](DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md)
freezes a contract-only committed-intent, integrity-finding, and terminal-summary
vocabulary. It does not add those records to `EpisodeOutcomeV2` or
`ExecutedRuntimeEvidenceV1`. `EpisodeOutcomeV2` cannot represent the protocol's
first-class integrity `ERROR` halt, and `ExecutedRuntimeEvidenceV1` cannot bind its
counter evidence. Additive outcome and executed-evidence successors are therefore
hard dependencies before runtime activation. Their names and shapes are not frozen
here; the existing schemas, vectors, hashes, readers, replay semantics, and Artifact
8 remain unchanged.

The now-frozen additive successor resources remain `contract_only`. Their executable
oracle establishes schema plus content-visible coherence: all section digests,
calendar/elapsed/event order, audit source prefixes, invocation coverage, and rejection
of attempts/decisions with an invocation later than a non-null finding invocation. It
does not prove absence of same-invocation or omitted post-finding work, nor authenticate
a coherently altered rollback/finality snapshot. Exact successor goldens separately pin
their five concrete fixtures. Runtime/replay provenance and no-post-finding execution
truth require the future authenticated execution verifier described above.

### 4.5 Build provenance and runtime identity adapters

`BuildProvenanceDescriptorV1` is frozen in
[build-provenance-descriptor-v1.schema.json](schemas/build-provenance-descriptor-v1.schema.json).
`source_commit` is evidence, never an eligibility constant. Eligibility is
content-addressed. Build mode is either explicit checkout mode or an audited wheel and
sdist pair with exact hashes and toolchain. Both bind exact `uv.lock` SHA-256.

`operatebench.clean_tracked_source_tree.v1` is a POSIX Git algorithm stated here in
full, so that two independent implementations produce the same digest and the same
refusal for the same repository. Its refusal vocabulary is closed and its conformance
vectors are machine-readable in
[clean-tracked-source-tree-v1.golden.json](schemas/clean-tracked-source-tree-v1.golden.json),
validated by its
[schema](schemas/clean-tracked-source-tree-v1.schema.json).

**Platform and preconditions.** This algorithm is POSIX-only; non-POSIX execution
refuses with `mode_observation_unavailable`. On POSIX, evaluate these checks in order:

1. `git diff-index --cached --quiet HEAD --` MUST exit zero, else
   `staged_index_drift`. The index is compared with `HEAD` before the working tree so
   a staged change is never reported as ordinary dirt.
2. `git status --porcelain=v1 -z --untracked-files=all` MUST emit zero bytes, else
   `dirty_tracked_content`. Thus every tracked/index/worktree status record and every
   untracked non-ignored path refuses. Ignored paths remain excluded by Git itself.
3. Read the effective Boolean values of `core.fileMode` and `core.symlinks`. An
   explicitly configured `false` for either refuses with
   `mode_observation_unavailable`; explicit `true` and unset are accepted because the
   POSIX Git effective default is enabled. No local configuration mutation is needed.

**Enumeration and sequence admission.** Read
`git ls-tree -r -z --full-tree HEAD` and consume records exactly as emitted; never
sort them. Mode `160000` refuses with `submodule_entry`; any mode other than `100644`,
`100755`, or `120000` refuses with `unsupported_mode`. Before any framing, the complete
entry sequence MUST already be strictly ascending by canonical UTF-8 path bytes.
Equal adjacent path bytes refuse with `duplicate_path`; a decreasing pair refuses
with `out_of_order`. This check applies equally to Git enumeration and to entries
supplied directly to a framing implementation.

**Path admission.** `-z` emits raw, unquoted path bytes. A path is admissible only if
it is non-empty and at most 4,096 bytes; decodes as strict UTF-8 with no surrogate;
is already exactly Unicode NFC (`normalize("NFC", path) == path`); contains no byte
below `0x20`, equal to `0x7F`, or a backslash; does not begin with `/`; splits on `/`
into no empty, `.`, or `..` component; and has no component whose ASCII A-Z bytes
mapped to a-z equal `.git`. Invalid lexical UTF-8 paths refuse with `unsafe_path`, and
a valid UTF-8 path that is not already NFC refuses with `non_canonical_path`.

Exact canonical path bytes MUST be unique. In addition, map only ASCII bytes `A`–`Z`
in each NFC UTF-8 path to `a`–`z`, leaving every other byte unchanged. Two distinct
paths with the same resulting key refuse with `portable_path_collision`. This is the
contract's portable ASCII-case collision rule, not locale-sensitive or
implementation-dependent Unicode casefold. Exact duplicates refuse with
`duplicate_path` before applying the distinct-path collision rule.

**Content bytes.** For `100644` and `100755` the content is the exact working-tree
file bytes read in binary. For `120000` the content is the exact symlink target bytes
returned by `readlink`, never the bytes of the target it names. Any entry whose
working-tree bytes cannot be read refuses with `unreadable_bytes`.

**Fixed framing.** Initialize SHA-256 with the ASCII bytes
`operatebench.clean_tracked_source_tree.v1` followed by one NUL. For each admitted
entry, in admitted sequence order, append exactly: the mode-byte length as an unsigned
8-byte big-endian integer; raw ASCII mode bytes; the path-byte length as an unsigned
8-byte big-endian integer; exact NFC UTF-8 path bytes; the blob-byte length as an
unsigned 8-byte big-endian integer; and exact blob bytes. There are no per-entry or
per-field separators and no decimal length fields. A length outside `0` through
`2^64 - 1` refuses with `frame_length_overflow`; an implementation MUST NOT substitute
a variable-width representation. The resulting lowercase hex SHA-256 is
`source_tree_digest_sha256`. An empty tree still hashes the 42-byte domain prefix to
`68f6186b2db6dc772dc66283dfa332d841dcdc93e80a200afe7e2ee725906f43`.

The closed refusal vocabulary is exactly `dirty_tracked_content`, `duplicate_path`,
`frame_length_overflow`, `mode_observation_unavailable`, `non_canonical_path`,
`out_of_order`, `portable_path_collision`, `staged_index_drift`, `submodule_entry`,
`unreadable_bytes`, `unsafe_path`, and `unsupported_mode`. The source commit can
change without changing eligibility when this descriptor's content identity and all
required bindings are satisfied.

The exact eight-owner map, authoritative source leaves, B1 card schema pointers,
future projector names, selected-run cross-bindings, projection pointers, dependency
bindings, support states, closed rejection vocabulary, staged status, and dependency
order are normative in
[the runtime identity adapter registry](schemas/runtime-identity-adapter-registry-v1.json),
validated by its [schema](schemas/runtime-identity-adapter-registry-v1.schema.json).
The independent [adapter census](schemas/runtime-identity-adapter-census-v1.json) and
its [schema](schemas/runtime-identity-adapter-census-v1.schema.json) cover every
pointer with its owner, source, source paths, consumer, nullability and cardinality.

Card projection pointers name the frozen B1 card surface exactly: the roots are
`/operation_card`, `/semantic_scenario_card` and `/variant_card`, and each pointer is
either an exact entry of
[the B1 identity-field census](schemas/identity-field-census-v1.registry.json) or a
proper prefix all of whose census descendants carry the same owner. A pointer that
resolves nowhere in that census, or that mixes owners, is not a projection of B1.

Ownership is isolated. A fact is owned by exactly one component, and
`authoritative_source_leaves` — named module-level symbols, methods, and repository
resources — are globally unique across the eight owners, so one source mutation moves
exactly one component identity. Where a component consumes a fact it does not own, it
declares a `dependency_bindings` entry naming the owning component and the exact
pointer it binds, never a second copy of the pointer. A binding is admissible only
toward a component this one already requires in
[the frozen B2 component registry](schemas/identity-component-registry-v1.json), and
the set of components a runtime-bundle owner binds is exactly the set of its direct
B2 requirements that are themselves runtime-bundle owners. The model-visible fields,
outcome tool schema, model protocol version, engine class, request mapping, and the
action/evidence/retrieval declarations are therefore owned once and bound, not
re-projected; the B2 requirement edges are expressed as bindings rather than as
duplicated ownership.

The owner choice is B1-compatible production projectors for operation,
semantic-scenario, and variant cards. A parallel direct-runtime identity partition is
rejected. The three named projector outputs are required future production facts and
are unavailable now; test helpers are never authority. Adapters load/construct all
authoritative sources, resolve the selected run in dependency order, recompute cards
and the complete manifest, and compare any artifact/manifest only as a candidate.
Generic `/fixture/*` B2 conformance projections are forbidden as runtime identity.

Only the two exact B3.1 Artifact-8 hashes in the registry are assigned
`full_lifecycle`, through the future internal lane registry. There is no outcome
inference, general arm default, or assignment for any other hash. Only the exact
`openai-gpt-5-6-luna` fake lane is authorized, jointly bound to its artifact, ledger,
and freeze-manifest hashes. Arbitrary injected agents remain unsupported.

Historical B3.1 fixtures deliberately lack source revision and a complete build
descriptor. The current checkout cannot retroactively supply what their historical
execution did not record. Consequently neither fixture currently yields a complete
runtime bundle/build identity. The deterministic-reference **fresh** lane is the only
initial implementation target, and only after production B1 projectors and a build
descriptor exist. The frozen fake lane is a future target blocked on its historical
build descriptor or an owner-approved equivalent; its arm and agent assignments do
not make it supported today.

### 4.6 Dependency, status, and non-goals

| Stage | B3.3 status |
|---|---|
| contract schemas, vectors, registry, census, prose | **frozen** |
| production B1 projectors | unimplemented |
| BuildProvenanceDescriptor production construction/validation | unimplemented |
| private runtime-to-manifest binding | unimplemented |
| immutable constituents/evidence/digest production types | unimplemented |
| `can_runtime_replay` semantics/runtime | unimplemented |
| evaluator compatibility | unimplemented |
| capability mint/consume | unimplemented |

The serial dependency is: this contract amendment; production B1 projectors and build
descriptor; private runtime-manifest binding; canonical immutable constituents,
evidence, and digest; runtime replay compatibility; evaluator compatibility; then
capability consumption. No compatibility result record or named `Decision` is frozen
or implemented by B3.3.

Non-goals are production runtime authority, package exports, Artifact 9, evaluator or
analysis behavior, provider/network/live execution, transport construction,
credentials, rewriting Artifact 8 or B3.1 manifests/fixtures, arbitrary agents,
general live-provider support, and declaring either historical lane fully supported.

## 5. Artifact 8 compatibility and selected architecture

Artifact 8 bytes and genuine deterministic/fake-transport bundles MUST be frozen
before the writer version moves. Artifact 8 is never converted, upgraded,
rewritten, or represented as Artifact 9.

| Artifact 8 situation | readable | schema-valid | exact-runtime-replayable | regradable |
|---|---:|---:|---:|---:|
| valid current-contract bundle, compatible historical runtime available, replay and chosen evaluator in one OS process/authority lifetime | yes | yes | yes | yes, after capability consumption |
| valid bundle, exact replay succeeds only in pinned external image/process | yes | yes | yes (detached report) | no: `cross_process_regrade_unavailable` |
| readable/schema-valid but historical runtime or complete decision/evidence binding unavailable | yes | yes | no | no |
| malformed or digest/identity-invalid | reader may parse bytes | no | no | no |
| exact replay diverges or bundle audit fails | yes when schema-valid | yes | no | no |

The one supported architecture is: load Artifact 8 without rewriting it; run the
pinned historical runtime adapter, exact replay, capability minting, and selected
compatible evaluator inside one OS process and one replay-authority lifetime. A
pinned external image MAY return only a detached replay report or a fully completed
grade produced inside that image. It MUST NOT export an authenticated outcome or
capability.

Fork, spawn, pickle, copy, deepcopy, filesystem, IPC, container, and serialized
round-trip MUST not transfer capability. Genuine same-process exact replay mints
one; failed or external replay mints zero. Replay/regrade constructs no provider
transport, reads no credential, and makes zero provider/socket calls.

## 6. Future compatibility predicates

The B2 registry retains the five names `can_runtime_replay`,
`can_evaluator_regrade`, `can_analysis_recompute`, `can_compare`, and `can_pool` as a
closed vocabulary. Vocabulary is not executable semantics. B3.3 freezes no result
record and no named `Decision`. Runtime replay compatibility may be specified only
after canonical binding/evidence exists; evaluator compatibility follows it; analysis,
comparison, and pooling remain with their canonical owners. Regrade success will not
imply comparability or poolability, and legacy `engine_version` is never an Artifact-9
comparison key.

## 7. `CorrectionEnvelopeV1`

A correction is append-only content-addressed provenance, not an edit to an
artifact. Its exact closed field set, required types, lexical bounds, and nested
record shapes are normative in
[the Correction Envelope v1 JSON Schema](schemas/correction-envelope-v1.schema.json).
Unknown fields are refused at every defined record boundary. Maps whose keys are
defined by an identity manifest or evaluator contract remain typed maps; they are
not extension points in the envelope.

The correction ID is random opaque identity only; it is not a content digest. The
literal domains, exact wrapper keys and projections, timestamp normalization,
processing sequence, and Ed25519 input bytes are normative in the
[Correction digest and signature contract v1](CORRECTION_DIGEST_CONTRACT.md).
Its domains are exactly
`operatebench.digest.correction.runtime_replay_report.v1`,
`operatebench.digest.correction.evaluator_grade.v1`,
`operatebench.digest.correction.signature_base.v1`,
`operatebench.digest.correction.envelope.v1`, and
`operatebench.digest.correction.chain_link.v1`; no alias or reuse is accepted.
Ed25519 signs the canonical UTF-8 bytes of the descriptor-bearing signature-base
wrapper—not digest bytes or hexadecimal text. The final envelope digest includes
the signature bytes and is the value used by a successor's
`parent_correction_digest`. Neither derived signature-base nor envelope digest is
stored inside the envelope, so the dependency graph is acyclic. The
[machine-readable golden vectors](schemas/correction-digest-v1.golden.json) are
normative byte/digest conformance evidence and include unsigned and deliberately
non-attesting signed-shape cases.

Validation requires original Artifact availability and digest equality; exact
report/grade digest equality; coherent old→new evaluator transition; parent
existence or null genesis; one atomic append per envelope; unique correction ID and
envelope digest; no cycles; configured fork policy; and one-time runtime capability
consumption. Capability is never serialized. Unsigned means locally consistent,
not attested.

The writer MUST resolve safe ancestors without symlinks, create exclusively with
no-follow semantics and mode `0600` inside operator-owned `0700` custody, refuse
overwrite, write complete canonical bytes, flush and fsync the file, fsync the
directory, and expose no partial envelope as committed.

## 8. Required tests-to-be

Phase B RED tests MUST precede implementation and cover:

- evaluator-only mutation moves evaluator content/dependent bundles, not runtime;
- runtime mutation moves runtime content/dependent bundles, not evaluator;
- card/scaffold/compiler/pricing/provider-setting mutations move only their owner
  and declared composites;
- unknown, unowned, multiply owned, nullable-required fields, domain collision,
  stale self-digest, missing dependency, and DAG cycle refusal;
- runtime replay without evaluator import/call and evaluator refusal of raw rows,
  caller outcomes, booleans, and serialized outcomes;
- capability one-shot behavior and fork/spawn/pickle/copy/deepcopy/filesystem/IPC
  non-transfer;
- genuine same-process Artifact 8 replay issues exactly one capability; failed or
  external replay issues none; all paths prove zero transport/credential/socket;
- byte-exact Artifact 8 fixtures remain unchanged after writer 9 ships;
- ledger 2 and Artifact 8 bundle audits remain valid using literal historical
  thresholds rather than `ARTIFACT_VERSION` aliases;
- chain swap, orphan, duplicate, cycle, configured fork, report substitution,
  grade substitution, nested mutation, symlink ancestor/final path, partial write,
  overwrite, durability, and signature-status separation.

## 9. Serial Phase B ownership and acceptance

The merged RFC numbering in this section is authoritative and supersedes stale
labels in earlier standalone plans. Phase B remains strictly
`B1 -> B2 -> B3 -> B4 -> B5`; a later stage MUST NOT
merge, publish evidence, or become a dependency of an earlier incomplete stage.

### B1 — card foundation, packaging, and census

B1 owns `src/operatebench/cards.py`; packaged copies of
`operation-card-v1.schema.json`, `semantic-scenario-card-v1.schema.json`,
`variant-card-v1.schema.json`, `card-defs-v1.schema.json`,
`card-registry-v1.json`, `identity-field-census-v1.schema.json`, and the expanded
`identity-field-census-v1.registry.json`; and package exports from
`src/operatebench/__init__.py`. It also owns
`tests/test_card_schemas.py`, `tests/test_card_identity.py`,
`tests/test_card_registry.py`, `tests/test_identity_field_census.py`, and packaging
assertions in `tests/test_operatebench_packaging.py`.

B1 acceptance requires offline `$id`/`$ref` resolution, Draft 2020-12
meta-validation, strict unknown-field and duplicate-key refusal, semantic
cross-record validators, canonical timestamp/calendar validation, digest vectors,
all card mutation boundaries, and a fully expanded zero-or-one-owner census of the
Card v1 surface with nonempty consumers. This is not the full Artifact 9 census;
B4 supplies that separately. `src/operatebench/identity.py`, an Identity Manifest,
and any Artifact 9 reader/writer MUST NOT import or depend on card implementation
until these B1 tests are green. B1 is therefore the Card implementation and Card
census owner, not an informal prerequisite delegated to B2.

### B2 — identity manifest and compatibility graph

B2 Slices 1–2 own the pure-offline `src/operatebench/identity.py` construction
kernel, the strict manifest schema, fixed component/composite registry, built-in
compatibility-contract registry, conformance vectors, component/composite
recomputation, and registry/DAG tests. The `identity.py` module itself has no direct
import of B1 cards or any Artifact, runtime, evaluator, provider, or version module,
and does not use constants from those modules. Normal Python package initialization
may load the package's existing B1 top-level exports before the submodule; B2 does
not change `src/operatebench/__init__.py` or add top-level exports.
For this PR boundary, B2 Slice 3 owns the manifest validator and
manifest-projection census, while Slice 4 owns only the bounded component-
requirement evaluator and frozen predicate vocabulary. B3 owns the semantics and
canonical inputs of both `can_runtime_replay` and `can_evaluator_regrade`; the
canonical analysis/result owner owns analysis recomputation, comparison, and
pooling. Slice 5 owns `identity_diff`/`OwnedMutation` and later domain projection
adapters. B4 owns only the Artifact 9 adapters/serialization, complete census, and
acceptance. The validator accepts a census explicitly, but the B1 Card-only census
is not a substitute and creates no backward dependency from B2 to B4.

### B3 — split replay/evaluation authority

B3 owns immutable executed runtime evidence, same-process single-use replay
capability, evaluator admission, Artifact 8 frozen fixtures, zero-provider replay,
and process/copy/IPC refusal tests. It owns the semantics and canonical inputs of
both `can_runtime_replay` and `can_evaluator_regrade`. It consumes B2's validated
manifest and does not write Artifact 9.

The additive contract-only successor pair is specified by
[AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md](AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md).
It does not widen EvidenceV1, authorize V2 shape as runtime evidence, construct an
identity, or change Artifact 8. The dedicated audit object is the sole Core witness;
its operation instance, accepted Operation Core digest, future Card aggregate/source
declaration, finding/summary, outcome disposition, provider dispatch, and both nested
and outer digests require future semantic verification.

### B4 — Artifact 9 and ledger migration

B4 owns the Artifact 9 schema/reader/writer, explicit historical ledger-binding
minimum, execution-ledger intended-version move, bundle audit, and byte-exact
Artifact 8 non-regression tests. Its compatibility work is limited to Artifact 9
adapters and serialization of B3-owned replay decisions; B4 MUST NOT redefine
runtime replay or evaluator regrade semantics or canonical inputs. The writer
version moves only after B1–B3 gates.

### B5 — corrections and release evidence

B5 owns correction digest/signature implementation, append-only custody and chain
validation, conformance against the golden vectors, correction mutation tests,
and release documentation. It consumes B4 artifacts and cannot retroactively edit
Artifact 8 or 9 bytes.

## 10. Phase boundaries

This RFC is the normative Phase A contract. Phase B acceptance evidence consists
of executable types, machine-readable ownership census, frozen Artifact 8 fixtures,
independent mutation tests, zero-provider runtime replay, Artifact 9 reader/writer,
and correction-chain tests. Until those exist, no baseline artifact is described
as Artifact 9 and no official regrade is available. The sealed Alpha remains
preserved and `UNINTERPRETABLE`; diagnostics do not replace it.
