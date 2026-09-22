# Grok-pair episode100 preparation (not paid authorization)

`tools.run_episode100` defaults to the unchanged six-cell selection. Its only
optional subset is `--cells grok45,grok46`, in that order. This selects controller
`synthetic-demo-episode100-grok-pair-v1`; the underlying episode100 agent,
specification, counting and settlement policy are unchanged. It starts one new
episode for each selected model, not six episodes and not a continuation of an
old journal. A provider failure ends that model's one attempt; no retry, tuning
or automatic replacement attempt is performed. The other model is independent.

The xAI Responses emitted schema spells all fifteen nonempty patterns as
`^[\s\S]+$`. Core's search-semantic `[\s\S]` and all non-pattern schema leaves
remain unchanged, including strict=false, null elision and array rules.
The schema-spelling correction advanced Grok 4.5 v3 to v4 and Grok 4.6 v4 to v5;
the subsequent compact action-schema projection uses v6 and v7 respectively.
The historical output-policy identity is not repinned. Always interpret mapping
identities together with the model, not as globally unique serial numbers.
Other providers and historical artifacts are not migrated.

Per-turn deadline stays 30 seconds. **The actual pinned source has 4096 output
tokens for Grok 4.5 and 8192 for Grok 4.6**, not 8192 for both. Effort remains low
and high respectively; sampling omissions and other controls are untouched.
Do not silently change a ceiling to match an assumption in launch prose.
Historical fixed-canary envelope estimates are not episode100 admission bounds:
the latter derives finite controls from the new cap and reserves actual wire
bounds. The current synthetic Grok 4.6 wire maximum is 27279, not the frozen
historical estimate 27002. No historical cap is raised by this entrypoint.

## Required new external series manifest

The pair requires `--series-manifest /ABS/NEW-SERIES.json`: owner-owned, canonical
absolute, regular single-link file, mode 0600. Exact keys:

- `schema`: `episode100-series-v1`
- `series_id`: nonempty owner-selected identifier
- `prior_evidence_sha256`: nonempty list of lowercase SHA-256 strings linking the
  immutable previous run/probe accounting evidence. These are references, not
  authority; the parent must independently verify the referenced files.
- `authorized_total_usd`: positive finite decimal string for the whole series
- `committed_exposure_usd`: positive finite decimal string, including permanently
  retained probe reservations; measured probe usage must not be added twice
- `new_cap_usd`: positive finite decimal string, equal to the CLI cap

Exact rational arithmetic enforces `authorized_total_usd = committed_exposure_usd
+ new_cap_usd`. The new ledger owns **only new_cap_usd**, never a fresh whole-series
ceiling. The manifest is read once before credentials, bound by its canonical
SHA-256 in new authority and the durable consumption receipt, then retained and
fsynced as `OUTPUT/series-manifest.json` before any dispatch. The summary repeats
its digest and committed carry. The carry is immutable launch evidence, not a
refundable reservation in the new ledger. Retain the manifest even after failure.

Canonical digest uses `tools.aggregate_budget.canonical`: sorted keys, compact
JSON separators, UTF-8, default ASCII escaping and one final newline, then SHA-256.
The source JSON need not have this whitespace; authority binds the parsed value.

## New approval schema and command — parent-owned, do not run during preparation

Use the existing episode100 owner-file gate in a **new** canonical owner-only
0700 authority directory, with a new 0600 regular single-link approval file.
Exact keys (no extras):

- `profile`: `synthetic-demo-episode100-grok-pair-v1`
- `source_sha`: exact verified clean checkout HEAD (40 hexadecimal characters)
- `output`: new canonical absolute output directory, not yet existing; its parent
  must exist outside the repository and be owner-controlled
- `cap_usd`: exact decimal string used on the command line
- `cells`: `["grok45", "grok46"]`
- `max_episode_decisions`: integer 100
- `retries`: integer 0
- `approved`: boolean true
- `historical_authorities_retired`: boolean true (parent must actually retire them)
- `mode`: `paid`
- `credential_file`: canonical absolute 0600 regular single-link JSON mapping
  **only** `grok45` and `grok46` to their keys
- `series_manifest_sha256`: canonical manifest digest described above

No real approval or credential file is supplied here. Source/scope/count/cap and
series binding must pass before credential custody, and O_EXCL consumption plus
fsync precede key reads. Old authorities/journals must never be retried. Failure
after consumption does not unconsume it or authorize another attempt.

Non-authorizing command template, from the approved clean checkout only:

```sh
env -i PATH=/usr/bin:/bin PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 "${EPISODE100_PYTHON:?absolute verified interpreter}" -m tools.run_episode100 --live --cells grok45,grok46 --series-manifest /ABS/NEW-SERIES.json --authority /ABS/NEW-OWNER-AUTHORITY/approval.json --output /ABS/NEW-OUTPUT --cap-usd NEW_REMAINING_USD
```

For safe rehearsal use `--offline` instead of `--live`, omit `--authority`, use a
synthetic series manifest, and keep output new. `--mock-authority` exercises the
same consuming gate with public placeholder keys over real SDK MockTransports.
Ordinary reference episodes complete at 46 decisions; separate negative tests
exercise 100-decision retained partial evidence, including a successful prefix
across invocations, without resetting at the former six-decision boundary.
