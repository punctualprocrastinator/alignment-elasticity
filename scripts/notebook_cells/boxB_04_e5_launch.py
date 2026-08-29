# Box B E5 second-family launch (marimo id LwcO). Detached worker: Qwen3-8B-Base
# refusal direction (diff-in-means + logistic tol=1e-10) at the proportional
# layer round(20/32 * n_layers)=22, carried unchanged to Qwen3-8B. Arms:
# base-neutral, instruct-neutral (format-matched primary), instruct-chat
# (own template, enable_thinking=False, deployment caveat). Per arm: margin,
# efficacy(|c|<=2)+CI, 20-random-dir null z, c50, behavioural refusal crossing.
# Then e5_family.e5_judge_launch() runs HarmBench-cls genuine-harm on all arms.
# Poll /marimo/e5/status.json + log.txt; pull e5_arm_*.json as they land.
print(e5_family.e5_launch())
