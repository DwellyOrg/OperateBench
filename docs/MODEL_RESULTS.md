# Model results — descriptive aggregate observations

One common list of 27 exact model IDs, including Gemini. These are descriptive observations across heterogeneous runtime/evaluator versions and provider settings, not a controlled ranking or official Lifecycle Track leaderboard. Independent human semantic validation remains pending.

The original 21 models retain the unweighted mean of two original scored-case rates; the additional six have one effective scored-case rate each. Exact rational quality, then goal achievement, then casefolded ID and exact ID determine order; percentages round only for display. Goal achievement includes strict-quality cases. Rates are conditional on scored coverage. Visible Scored coverage totals scored/assigned across available observations; rates are never pooled across cases. Each observation assigned 44 related synthetic cases, one trial per case. Technical exclusions and interrupted aborts are not quality failures and do not enter scored denominators. The original 21 models and 162 original scored records retain their grades. Linked technical corrections replace only excluded or aborted cells within the same observation: 207 scored, 57 technical exclusions and zero aborted out of 264 assigned; 106 correction attempts and all 370 dispositions are retained.

Compliance 0.3.0 applies to the additional observations; historical grades are frozen. Mixed original and correction source provenance is in the [approved selected-six aggregate](studies/2026-10-selected-six.json); the top-level correction revision is a provenance anchor, not a claim that all records ran on that revision. Attempt-group hashes describe admitted-attempt contracts, not observed wire controls or uniform native execution. Later publication changes are not execution provenance. The [machine means](studies/2026-10-model-averages.json) match the site. Six new costs remain Unknown/null; retained estimates and owner cost scenarios are not invoices or provider attestations. Raw traces and usage remain private.

The [original public revision](https://github.com/DwellyOrg/OperateBench/blob/f3bd45f/docs/MODEL_RESULTS.md) preserves all 19 historical observations. The [21-model, two-series study](studies/2026-10-observations.md) and its seven historical artifacts remain unchanged. This bounded owner-directed aggregate extension does not authorize future results or official release; human semantic validation remains pending.

| Model | Strict quality | Goal achieved | Estimated run cost (USD, not invoice) | Scored / assigned |
| --- | --- | --- | --- | --- |
| `gpt-6-astra` | 96.6% | 100.0% | $238.13 | Scored 87/88 |
| `gemini-4-argon` | 95.5% | 100.0% | $94.57 | Scored 88/88 |
| `gpt-6.1-sol` | 95.5% | 100.0% | $52.67 | Scored 88/88 |
| `gpt-6.1-terra` | 86.4% | 100.0% | Unknown | Scored 44/44 |
| `claude-fable-5-1` | 80.5% | 100.0% | $192.63 | Scored 87/88 |
| `claude-opus-5-5` | 54.5% | 98.9% | $153.68 | Scored 88/88 |
| `gemini-3.8-flash` | 45.3% | 100.0% | $29.09 | Scored 86/88 |
| `gpt-6-sol` | 44.3% | 89.8% | $74.55 | Scored 88/88 |
| `grok-4.7` | 33.3% | 100.0% | Unknown | Scored 15/44 |
| `muse-wonder-1.0` | 26.5% | 56.4% | $52.06 | Scored 83/88 |
| `GLM-5.3` | 16.7% | 88.9% | Unknown | Scored 36/44 |
| `zai-org/GLM-5.3-Flash` | 14.8% | 77.3% | $1.21 | Scored 88/88 |
| `nvidia/GLM-5.2-NVFP4-262K` | 13.6% | 71.6% | $1.73 | Scored 88/88 |
| `muse-spark-1.3` | 10.2% | 42.0% | $46.82 | Scored 88/88 |
| `Qwen/Qwen3.8-27B-FP8` | 3.4% | 3.4% | $2.52 | Scored 88/88 |
| `deepseek-ai/DeepSeek-V4.1-Flash` | 2.3% | 18.4% | $2.90 | Scored 87/88 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 2.3% | 5.7% | $2.54 | Scored 88/88 |
| `moonshotai/Kimi-K3` | 0.0% | 43.2% | Unknown | Scored 44/44 |
| `gpt-6-luna` | 0.0% | 21.5% | $1.42 | Scored 84/88 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 0.0% | 17.0% | $1.96 | Scored 88/88 |
| `nvidia/Kimi-K2.6-NVFP4` | 0.0% | 16.1% | $1.00 | Scored 87/88 |
| `Qwen/Qwen3.5-397B-A17B-FP8` | 0.0% | 6.8% | $1.36 | Scored 88/88 |
| `claude-haiku-5-5` | 0.0% | 0.0% | Unknown | Scored 44/44 |
| `mistral-large-4` | 0.0% | 0.0% | Unknown | Scored 24/44 |
| `mistral-medium-3-5` | 0.0% | 0.0% | $25.77 | Scored 88/88 |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 0.0% | 0.0% | $2.26 | Scored 88/88 |
| `Qwen/Qwen3.5-9B` | 0.0% | 0.0% | $1.27 | Scored 88/88 |
