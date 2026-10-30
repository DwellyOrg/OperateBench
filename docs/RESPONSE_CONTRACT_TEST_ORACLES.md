# Response contract test oracles

The extension primitives goldens independently state the current OpenAI server
metadata v4 contract (null or closed cyber/standard access metadata) and the
Lifecycle Mistral contract (closed reasoning-token details and token subsets).
Their digests are computed from those literal documents, never substituted for
the documents. Each module/constant is tested separately so an earlier mismatch
cannot hide another provider's drift.

The OpenAI composition tests retain the original request bytes, adapter identity,
and settings baseline. They explicitly reverse only the reviewed response
interpretation fields before checking the original settings digest. The
Lifecycle Mistral settings snapshot pins the complete current mapping, including
request fields, output ceiling, retry policy, SDK identity, and tool names.
Boundary Mistral and all xAI literal contracts retain their prior values.

OpenAI already advances its server metadata identifier to v4. Lifecycle Mistral
retains its extension family identifier v1 with a changed content digest. Its
settings carry both: the digest distinguishes the accepted response contract
and therefore run configuration even when the family label stays the same.
No production identifier correction is required for this test-oracle repair.
These snapshots neither rewrite historical identities nor establish equivalence
between runs made under different response interpretations.

Legacy null-metadata transport fakes explicitly select the legacy branch with
`selected_six=False`. Selected-six fixtures retain their own branch and strict
usage agreement checks. All null, non-null, unknown-field, and raw/typed
adversarial expectations remain enforced.

This repair changes tests and documentation only. It changes no generation
settings, request controls, budget, case, evaluator, grade, or stored record.
