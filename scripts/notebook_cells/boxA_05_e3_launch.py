# E3 layer robustness: refit base diff-in-means direction at layers {8,12,16,20,24,28},
# compute efficacy+margin for base/sft-1000/instruct at each layer. Reduced c-grid (|c|<=2).
import e3_layer as e3
print(e3.e3_launch())
