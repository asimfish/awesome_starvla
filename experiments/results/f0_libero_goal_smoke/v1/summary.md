| run | logged pts | loss steps 1-50 | loss last 50 | in-train MSE (last) | s/step | per-head loss (last 50) |
|---|---:|---:|---:|---:|---:|---|
| f0_oft | 30 | 0.574 | 0.231 | 0.0157 | 2.14 | - |
| f0_multihead | 30 | 2.386 | 1.241 | 0.0168 | 2.76 | - |

| run | probe step | drift mean | drift max | max layer |
|---|---:|---:|---:|---|
| f0_oft | 0 | 0.0000 | 0.0000 | layer_0 |
| f0_oft | 50 | 0.0593 | 0.1535 | layer_22 |
| f0_oft | 100 | 0.0110 | 0.0325 | layer_19 |
| f0_oft | 150 | 0.0363 | 0.0949 | layer_19 |
| f0_oft | 200 | 0.0426 | 0.1037 | layer_22 |
| f0_oft | 250 | 0.0085 | 0.0238 | layer_25 |
| f0_multihead | 0 | 0.0000 | 0.0000 | layer_0 |
| f0_multihead | 50 | 0.0332 | 0.0793 | layer_16 |
| f0_multihead | 100 | 0.0490 | 0.1166 | layer_16 |
| f0_multihead | 150 | 0.0687 | 0.1605 | layer_16 |
| f0_multihead | 200 | 0.0625 | 0.1458 | layer_16 |
| f0_multihead | 250 | 0.0027 | 0.0104 | layer_33 |
