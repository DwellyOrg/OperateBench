# Source-changing technical corrections

The optional `correction_binding` admission extension uses schema
`three-flow-correction-binding-v1`. It authorizes execution under the current
reviewed source/runtime/registry, **not a replacement financial policy**.
Absent this extension, existing admission/receipt/genesis serialization is unchanged.
Model-mode selection remains independent and retains its existing default.
The optional `mode_profile: off-or-minimum-v1` and
`transport_policy: paced-safe-errors-v1` fields bind the current execution
registry exactly. The latter selects process-wide Mistral pacing only; other
providers retain their existing transport policy. Both settings are forwarded
to the real SDK transport factory. Omitting either retains its legacy default,
not permission to drop a setting from an already-bound registry.

The embedded original admission retains its original mode and transport fields
and structurally validated historical registry; it is not compared with today's
settings. Its consumed closed receipt authenticates the complete original
admission. New-series pricing binds the new series source and selected registry;
correction pricing instead remains identical to the original admission, even
when the current execution mode or pacing changes.

The closed binding contains explicit `owner_accepted`, `owner_evidence_sha256`,
`original_source_sha`, `execution_source_sha`, `original_admission`,
`financial_policy_sha256`, `genesis_sha256`, `manifest_sha256`, and
`original_trial_artifacts_sha256`. The latter maps the exact selected original
trial IDs to `assigned`, `binding`, and `closure` file-byte SHA256 values; these
must exist unchanged before recovery or credential access.
The financial digest uses admission `digest` on exactly `cap_usd` and
`limit_policy` (null for the historical capped controller). Genesis hashes the
complete first canonical journal line, including newline; manifest hashes the
unchanged `assignment.json` bytes. The original admission must hash to the
original consumed receipt's `admission_sha256`. Retain that external document;
never reconstruct it by inventing financial consent.

Existing admission fields additionally bind campaign/custody roots, namespace,
original consumed claim, current journal digest, current runtime/registry,
distinct attempt, ordered subset, and confirmed technical-fix evidence. The
consuming path verifies the original receipt before opening the existing budget;
the journal and genesis hashes are checked under its exclusive lock before
recovery. Every prior settled/unknown reservation survives; unresolved exposure
is conservatively converted to unknown by existing recovery, never refunded.

Selected profiles must remain exactly equal to the original admitted profiles.
Owner pricing stays verified against its **original** source and registry entry;
the outer admission checks the **new** execution source/runtime/registry.
This extension does not authorize price upgrades or changed financial assumptions.
Such changes require separately designed explicit consent, not editing this
binding or the immutable genesis. Historical liability references stay unchanged.

Only correction operations can carry the extension. A changed-source correction
without it refuses, including the capped legacy controller. Existing same-source
operations keep their existing behavior. Technical eligibility, subset scope,
new correction IDs, cap exhaustion, and one-shot durable claims still apply.
Consumption precedes credential reads; rejection never creates a fresh wallet.
No approval writer or recovery command is supplied here. Hashes are integrity
bindings, not signatures: external owner custody and approval remain required.

Tests use disposable private synthetic custody and actual SDK mock transports;
none of this authorizes re-opening a real journal or launching a paid correction.
