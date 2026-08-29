# Box B E5-extra (Llama + Gemma via unsloth full-precision mirrors; official
# Meta/Google repos license-gated, user-approved). Ran as detached scratchpad
# workers, mirrored here. Reuses e5_family functions + e2_behavioural harness.
# verify_fp asserts bf16 + expected param count + no quantization_config before
# fitting (a quantized load STOPS that family). Proportional layer: Llama L20
# (32 layers), Gemma L26 (round(20/32*42)). gemma loaded attn_implementation=eager.
# import e5_extra
# e5_extra.run_family_launch("llama"); ... pull arms ... ; e5_extra.judge_launch("llama")
# e5_extra.run_family_launch("gemma"); ... pull arms ... ; e5_extra.judge_launch("gemma")
# Results: results/E5_llama.json, results/E5_gemma.json + fig_E5_{llama,gemma}.png
print("E5-extra: see e5_llama_*.json / e5_gemma_*.json + E5_{llama,gemma}.json")
