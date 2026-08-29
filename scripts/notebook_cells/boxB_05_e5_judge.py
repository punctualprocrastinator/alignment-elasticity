# Box B E5 HarmBench judge (marimo id AwQo). Phase 2: loads
# cais/HarmBench-Llama-2-13b-cls once and judges unsteered(c=0) vs steered(c50)
# generations for all three arms (base-neutral, instruct-neutral, instruct-chat),
# then re-aggregates E5_second_family.json. Reuses e2_behavioural harness
# (sentencepiece cache_clear + use_fast=False workaround).
print(e5_family.e5_judge_launch())
