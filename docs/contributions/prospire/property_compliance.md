<!-- SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Synthetic property-compliance partner pattern

'lettings.property_compliance.synthetic' is a maintainer-authored reference
showing the contribution shape available to a partner. It is an incubator,
fully invented, and not evidence-eligible. It is not an independently authored
external-partner contribution, a model result, public benchmark case, real
Dwelly workflow or policy, UK legal advice, or a statement of actual compliance
requirements.

## Scaffold to registration

Historical record only — this exact command is not runnable now that the pack ID
is registered:

```console
operatebench init-operation --pack-id lettings.property_compliance.synthetic --operation-id lettings_property_compliance_synthetic_v1 --owner-id prospire --owner-display-name "PROSPIRE TECHNOLOGIES LTD" --contribution-kind maintainer --destination src/operatebench/contributions/prospire/property_compliance
```

A new partner uses the guided repository-aware command and replaces the
fictional identity consistently:

```console
operatebench init-company-operation --repository-root . --owner-id partner_name --owner-display-name "Example Partner Ltd" --operation-name new_operation --pack-id partner.partner_name.new_operation.synthetic --operation-id partner_name_new_operation_synthetic_v1
```

The generated TODO sentinel was run and observed failing. The scaffold was then
completed vertically: strict fixture loading; V1 normal renewal; V2 invented
remediation and reinspection; V3 no-access expiry transfer; seven targeted
negative controls; deterministic serialization and fresh replay; finally static
registration. The only dispatch integration is the reviewed PACK import and
tuple entry in 'operatebench.sdk.builtins'; specs cannot select executable code.

The package owns its spec loader, state, reducer, fixed agents, evaluator,
negative-control oracle, run serializer, replay recomputation and SDK adapter.
Partners should copy that ownership boundary, synthetic disclaimers, exact-field
validation, authority separation, guarded finality, paired oracle pass/fail
expectations and offline tests. They should not copy this invented regime as
policy, reuse its causal details for production, import another domain's runner
or evaluator, modify Core/SDK dispatch, add dynamic loading, or claim evidence
status.

## Offline commands

```console
operatebench list-packs
operatebench validate --pack lettings.property_compliance.synthetic examples/contributions/prospire/property_compliance/operation.yaml
operatebench run --pack lettings.property_compliance.synthetic --spec examples/contributions/prospire/property_compliance/operation.yaml --scenario V1 --agent reference --output /var/tmp/property-v1.json
operatebench replay --pack lettings.property_compliance.synthetic --spec examples/contributions/prospire/property_compliance/operation.yaml --run /var/tmp/property-v1.json
operatebench check --pack lettings.property_compliance.synthetic --spec examples/contributions/prospire/property_compliance/operation.yaml
```

Repeat run and replay with V2 and V3 using distinct output paths. Outputs are
exclusive and never overwritten. Replay executes fresh pack semantics and
compares the complete recomputed record; it does not trust recorded self-checks.
