# Model results — exploratory retrospective table

Observed results for all 28 distinct exact model IDs tested across separate
campaigns, sorted alphabetically by ID. Each column retains its own cohort and
scoring denominator; there is no pooled score or overall rank.

UK_NORMAL covers 17 one-episode attempts. Maintenance V1 covers 17 one-episode
attempts: 14 fresh and 3 renewed, labelled in each cell. Earlier V1 diagnostics
cover 6 one-episode attempts: 3 scored incomplete and 3 excluded. Meta covers
44 assigned synthetic cases per model, executed on 2026-10-13 UTC; Wonder's
V1 case within the Meta 44-case maintenance flow is excluded, leaving 43 scored
cases, while Spark has 44.

The exact Meta paid execution source is [this public commit](https://github.com/DwellyOrg/OperateBench/commit/c6fb49bf2a3714c094c4).

`—` = untested in that cohort; `excluded` = unscored due to the stated adapter,
budget or provider condition; `operator-stopped` = unscored after an operator stop.
Goal completion and strict reliability are separate evaluator outcomes.

| Exact model ID | UK_NORMAL (1 episode) | Maintenance V1 (1 episode; fresh/renewed) | Earlier V1 diagnostic (1 episode) | Meta same-44 (scored denominator) |
| --- | --- | --- | --- | --- |
| `claude-fable-5-1` | goal yes · strict yes | goal yes · strict no (fresh) | — | — |
| `claude-haiku-4-5-20251001` | — | — | excluded (budget) | — |
| `claude-opus-5-5` | goal yes · strict yes | — | — | — |
| `claude-sonnet-5` | — | — | goal no · strict no (scored incomplete) | — |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | goal no · strict no | goal yes · strict no (fresh) | — | — |
| `deepseek-ai/DeepSeek-V4.1-Flash` | excluded (adapter) | goal yes · strict no (fresh) | — | — |
| `gpt-5.6-luna` | — | — | goal no · strict no (scored incomplete) | — |
| `gpt-6-astra` | goal yes · strict yes | goal yes · strict yes (fresh) | — | — |
| `gpt-6-luna` | goal yes · strict no | — | — | — |
| `gpt-6-sol` | goal yes · strict no | — | — | — |
| `grok-4.5` | — | — | excluded (provider) | — |
| `grok-4.6` | — | — | excluded (budget) | — |
| `grok-4.7` | — | goal yes · strict no (renewed) | — | — |
| `mistral-medium-3-5` | goal no · strict no | goal no · strict no (fresh) | — | — |
| `mistral-small-2603` | — | — | goal no · strict no (scored incomplete) | — |
| `muse-spark-1.3` | — | — | — | goal 18/44 · strict 4/44 |
| `muse-wonder-1.0` | — | — | — | goal 27/43 · strict 12/43 (1 excluded) |
| `nvidia/GLM-5.2-NVFP4` | — | goal yes · strict no (renewed) | — | — |
| `nvidia/GLM-5.2-NVFP4-262K` | goal yes · strict no | goal yes · strict no (fresh) | — | — |
| `nvidia/Kimi-K2.6-NVFP4` | goal no · strict no | goal yes · strict no (fresh) | — | — |
| `nvidia/Kimi-K2.6-NVFP4-262K` | — | goal no · strict no (renewed) | — | — |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | goal no · strict no | goal no · strict no (fresh) | — | — |
| `Qwen/Qwen3.5-122B-A10B-FP8` | operator-stopped | goal no · strict no (fresh) | — | — |
| `Qwen/Qwen3.5-397B-A17B-FP8` | goal no · strict no | goal yes · strict no (fresh) | — | — |
| `Qwen/Qwen3.5-9B` | goal no · strict no | goal no · strict no (fresh) | — | — |
| `Qwen/Qwen3.6-35B-A3B-FP8` | excluded (adapter) | goal no · strict no (fresh) | — | — |
| `Qwen/Qwen3.8-27B-FP8` | goal no · strict no | goal yes · strict no (fresh) | — | — |
| `zai-org/GLM-5.3-Flash` | goal yes · strict no | goal yes · strict no (fresh) | — | — |

- Cohorts use different cases, execution conditions and scoring eligibility.
  One-case results and the Meta series cannot support a statistically stable
  overall ranking or a production recommendation.
- Fresh and renewed Maintenance attempts remain distinct historical cohorts;
  earlier diagnostics retain their original scored-incomplete or excluded status.
  Exclusions and operator stops are not quality failures.
- A separate later `grok-4.6` episode100 completed with strict = no in 51 turns
  at approximately USD 1.04103. It does not replace the earlier budget exclusion.
- Meta usage-based published estimates are USD 32.801164 for Wonder and
  USD 11.28384925 for Spark, totaling USD 44.08501325. These are not invoices
  and are not comparable to older owner budget envelopes.
- Independent human validation is pending. Single trials and related synthetic
  variants do not establish repeatability, construct validity or general capability.

Results were transcribed from a verified sanitized aggregate summary, with no
new model calls, relabelling or regrading. Older cohorts predate the Meta series;
no common execution date is asserted. Raw trajectories are not published, so
this table alone does not provide independent reproduction of the runs. The
[publication manifest](../PUBLICATION_MANIFEST.json) remains a frozen historical record.

Public implementation: [runner](../src/operatebench/runner.py),
[Maintenance evaluator](../src/operatebench/domains/lettings/maintenance/evaluator.py),
[provider integrations](../src/operatebench/providers/), and
[Meta snapshot method and tariff](META_MUSE_SNAPSHOT.md).
