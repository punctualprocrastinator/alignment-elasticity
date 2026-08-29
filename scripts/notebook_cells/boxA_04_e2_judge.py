# E2 phase 2: HarmBench-cls validity check on a 20-prompt subset per
# (model, coeff). Run AFTER phase 1 completes; loads the cls model once.
import e2_behavioural as e2
print(e2.e2_judge_launch())
