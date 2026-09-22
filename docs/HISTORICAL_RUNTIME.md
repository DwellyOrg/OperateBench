# Historical mock runtime

The source distribution includes an audited dependency closure in
`tests/fixtures/historical_engine_0_9/source`. Its semantic Python modules and
resources retain their original bytes and engine 0.9.0 identity. This is test-only
mock regeneration, never a new real historical run or current-engine substitution.

`manifest.json` binds each included file by SHA-256. The harness independently
pins that manifest; `tests/historical_source_pins.py` additionally freezes every
individual file digest outside the manifest. Missing, extra, symlinked or changed
source refuses before subprocess execution. The subprocess denies socket access
and receives an environment allowlist with no inherited credentials.

Only the packaging/bootstrap boundary changed: content identity replaces the
private Git-object locator. The 40-character synthetic revision field used by
mock Alpha results is derived from the content manifest; it is not a Git commit.
No historical artifact is relabelled or repinned. B3 regeneration compares the
unchanged frozen bytes, and Alpha audits retain current static validation plus
historical causal checks with identical validator source. Source custody is
recorded outside the distributable project. No private source repository is needed.

The conservative closure contains transitive imports, resource-package data and
the authored maintenance fixture and lockfile needed by regeneration. It is not
a complete predecessor checkout. The wheel excludes this test-only source; the
sdist includes it under tests. Code retains Apache-2.0 licensing; the included
example fixture retains CC-BY-4.0. Rights and publication approval remain human gates.
