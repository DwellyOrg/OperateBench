# Correction digest and signature contract v1

**Status:** Phase A normative wire and cryptographic contract
**Schema:** [Correction Envelope v1](schemas/correction-envelope-v1.schema.json)
**Golden vectors:** [Correction digest vectors v1](schemas/correction-digest-v1.golden.json)

This document closes every byte-level choice for `CorrectionEnvelopeV1`. An
implementation MUST use these literals, projections, sequence, and bytes exactly.
A different literal, key set, projection, timestamp spelling, canonicalizer, or
signing input is a different protocol version, not a compatible implementation.

## 1. Canonical bytes and timestamp

`CJ(x)` is UTF-8 encoding of JSON with object keys sorted by Unicode code point,
array order retained, no insignificant whitespace, and the shortest JSON string
escapes (lowercase hexadecimal in `\u` escapes). Inputs refuse duplicate object
keys, floats, NaN/infinity, integers outside signed 64-bit, lone surrogates, and
non-string map keys. Set-class arrays are duplicate-free and sorted before `CJ`;
all other arrays retain their schema-declared order.

Every correction timestamp has the sole lexical form
`YYYY-MM-DDThh:mm:ss.ffffffZ`: UTC `Z`, exactly six decimal fractional digits, no
offset, leap second, or alternate equivalent spelling. JSON Schema enforces the
lexical form; the validator MUST also reject impossible Gregorian dates.

`H(x) = SHA-256(x)`. A digest field is 64 lowercase hexadecimal characters.

## 2. Frozen domains and wrapper projections

The literal domains are exactly:

| Identity | Literal domain | Canonical wrapper (exact keys) |
|---|---|---|
| replay report | `operatebench.digest.correction.runtime_replay_report.v1` | `{"domain": DOMAIN, "runtime_replay_report": REPORT}` |
| evaluator grade | `operatebench.digest.correction.evaluator_grade.v1` | `{"domain": DOMAIN, "grade": GRADE_WITHOUT_SELF_DIGEST}` |
| signature base | `operatebench.digest.correction.signature_base.v1` | `{"domain": DOMAIN, "envelope": SIGNATURE_BASE_PROJECTION}` |
| final envelope | `operatebench.digest.correction.envelope.v1` | `{"domain": DOMAIN, "envelope": COMPLETE_ENVELOPE}` |
| chain link | `operatebench.digest.correction.chain_link.v1` | `{"domain": DOMAIN, "parent_correction_digest": PARENT_OR_NULL, "correction_id": CORRECTION_ID, "envelope_digest": ENVELOPE_DIGEST}` |

Spaces shown above aid reading and are absent from `CJ`. Canonical key sorting,
not table order or construction insertion order, determines bytes.

`GRADE_WITHOUT_SELF_DIGEST` is `new_grade` with its one
`grade_digest_sha256` member omitted. Every other nested member is retained.
`SIGNATURE_BASE_PROJECTION` has the envelope schema's field order for processing
only; canonical bytes still sort keys. Its exact closed member set is the key set
of each machine-readable `signature_base.projection` in the normative golden-vector
file; unknown or missing keys are refused. All members other than the derived
signature descriptor are copied without transformation from the envelope.
`external_signature_descriptor` is `null` when `external_signature` is null;
otherwise it is the exact closed object `{schema_version, algorithm, key_id}`.
The `signature_base64` member is never present in this projection. The projection
does not omit timestamps or provenance. There is no `signature_base_digest` or
`envelope_digest` field in an envelope, so neither can create a self-cycle.

## 3. Mandatory sequencing and signed bytes

A writer and validator perform these steps in order:

1. Validate the closed envelope shape and canonical timestamp.
2. Compute `runtime_replay_report_digest = hex(H(CJ(report wrapper)))` and require
   exact equality with the field.
3. Compute `new_grade.grade_digest_sha256 = new_grade_digest =
   hex(H(CJ(grade wrapper)))` and require both exact equalities.
4. Construct the signature-base projection and wrapper. Compute the diagnostic
   `signature_base_digest = hex(H(CJ(signature-base wrapper)))`.
5. For `algorithm = "ed25519"`, sign **the complete `CJ(signature-base wrapper)`
   UTF-8 byte string**, not the 32 digest bytes, not its hexadecimal text, and not
   the bare projection. Encode the 64-byte signature as canonical padded standard
   Base64. An unsigned envelope uses `external_signature: null` and signs nothing.
6. Compute `envelope_digest = hex(H(CJ(envelope wrapper)))` over the complete final
   envelope, including `signature_base64` when present.
7. Compute `chain_link_digest = hex(H(CJ(chain-link wrapper)))`. A successor stores
   the predecessor's final `envelope_digest` in `parent_correction_digest`; it does
   not store `chain_link_digest` there.

Signature verification reconstructs the same descriptor-bearing signature-base
projection and verifies the decoded signature against those exact canonical UTF-8
bytes. Local digest validity and external signature validity are separate verdicts.
A signed-shape golden vector with `signature_verification_expected: false` tests
projection and envelope hashing only and MUST NOT be accepted as an attestation.

## 4. Golden-vector conformance

The machine-readable vector file stores each input envelope, every exact projection,
each canonical wrapper as both JSON text and UTF-8 hexadecimal bytes, and every
expected SHA-256 result. `unsigned` exercises the null descriptor. `signed_shape`
uses a documented all-zero 64-byte shape placeholder and deliberately expects
signature verification to fail; no private key or purported real signature is
invented. A conforming implementation MUST recompute every byte and digest and
MUST reject any one-byte mutation not accompanied by the corresponding downstream
digest changes.
