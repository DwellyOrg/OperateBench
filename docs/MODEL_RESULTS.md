# Model results — descriptive full44 comparison

19 exact model versions, each assigned 44 cases. Goal achieved and all quality
checks (strict) are separate evaluator outcomes; denominators count scored cases.

## Original 16-model campaign

704 assigned = 698 scored + 2 excluded + 4 aborted. Ordered by strict fraction,
then goal achieved. [Historical execution source](https://github.com/DwellyOrg/OperateBench/commit/58b36826894c4306aa3ab).

### GPT-6 Astra (`gpt-6-astra`)

Goal achieved: 44/44\
All quality checks: 42/44 · 95.5%\
Estimated cost*: $239

### Claude Fable 5.1 (`claude-fable-5-1`)

Goal achieved: 43/43\
All quality checks: 35/43 · 81.4%\
Estimated cost*: $189

### Claude Opus 5.5 (`claude-opus-5-5`)

Goal achieved: 43/44\
All quality checks: 22/44 · 50.0%\
Estimated cost*: $155

### GPT-6 Sol (`gpt-6-sol`)

Goal achieved: 39/44\
All quality checks: 19/44 · 43.2%\
Estimated cost*: $72

### GLM 5.2 NVFP4 262K (`nvidia/GLM-5.2-NVFP4-262K`)

Goal achieved: 34/44\
All quality checks: 6/44 · 13.6%\
Estimated cost*: $123

### GLM 5.3 Flash (`zai-org/GLM-5.3-Flash`)

Goal achieved: 33/44\
All quality checks: 4/44 · 9.1%\
Estimated cost*: $94

### DeepSeek V4 Flash 0731 (`deepseek-ai/DeepSeek-V4-Flash-0731`)

Goal achieved: 2/44\
All quality checks: 1/44 · 2.3%\
Estimated cost*: $213

### Qwen 3.8 27B FP8 (`Qwen/Qwen3.8-27B-FP8`)

Goal achieved: 1/44\
All quality checks: 1/44 · 2.3%\
Estimated cost*: $207

### GPT-6 Luna (`gpt-6-luna`)

Goal achieved: 9/40\
All quality checks: 0/40 · 0.0%\
Estimated cost*: $124

### DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash`)

Goal achieved: 9/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $221

### Qwen 3.6 35B A3B FP8 (`Qwen/Qwen3.6-35B-A3B-FP8`)

Goal achieved: 8/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $141

### Kimi K2.6 NVFP4 (`nvidia/Kimi-K2.6-NVFP4`)

Goal achieved: 7/43\
All quality checks: 0/43 · 0.0%\
Estimated cost*: $75

### Qwen 3.5 397B A17B FP8 (`Qwen/Qwen3.5-397B-A17B-FP8`)

Goal achieved: 3/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $110

### Mistral Medium 3.5 (`mistral-medium-3-5`)

Goal achieved: 0/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $164

### NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4 (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`)

Goal achieved: 0/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $185

### Qwen 3.5 9B (`Qwen/Qwen3.5-9B`)

Goal achieved: 0/44\
All quality checks: 0/44 · 0.0%\
Estimated cost*: $96

## Separately completed successors

Separate sources and settings; provider-specific `tool_choice=auto`. Sol has
44/44 scored; Wonder has 43/44; Spark has 44/44.
[Meta execution source](https://github.com/DwellyOrg/OperateBench/commit/c6fb49bf2a3714c094c4) ·
[Meta snapshot method and tariff](META_MUSE_SNAPSHOT.md).

### GPT-6.1 Sol (`gpt-6.1-sol`)

Goal achieved: 44/44\
All quality checks: 42/44 · 95.5%\
Estimated cost*: $53

### Meta Muse Wonder 1.0 (`muse-wonder-1.0`)

Goal achieved: 27/43\
All quality checks: 12/43 · 27.9%\
Estimated cost*: $33

### Meta Muse Spark 1.3 (`muse-spark-1.3`)

Goal achieved: 18/44\
All quality checks: 4/44 · 9.1%\
Estimated cost*: $11

*Costs are rounded whole-USD estimates, not invoices or a controlled cross-provider
cost comparison. Historical and Sol costs use conservative owner usage envelopes;
unknown cost exposure remains for Fable ($7.65), Kimi ($0.32) and Luna ($2.90).
Meta uses its separate tariff calculation. Fable and Kimi each have one excluded
case; Luna has four aborted cases; Wonder has one excluded case. These unscored
cases are not scored failures. Related synthetic variants, one trial per case,
and no independent human review (validation pending) limit these descriptive
results; they support no general capability claim. No new model calls or regrading.
