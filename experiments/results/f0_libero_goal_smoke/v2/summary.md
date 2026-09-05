| run | logged pts | loss steps 1-50 | loss last 50 | in-train MSE (last) | s/step | per-head loss (last 50) |
|---|---:|---:|---:|---:|---:|---|
| f0v2_oft | 30 | 0.628 | 0.244 | 0.0168 | 1.86 | - |
| f0v2_multihead | 30 | 2.493 | 0.998 | 0.0169 | 2.70 | gr00t=0.432, oft=0.243, pi=0.323 |

| run | probe step | drift mean | drift max | max layer |
|---|---:|---:|---:|---|
| f0v2_oft | 0 | 0.0000 | 0.0000 | layer_0 |
| f0v2_oft | 25 | 0.0133 | 0.0348 | layer_17 |
| f0v2_oft | 50 | 0.0205 | 0.0526 | layer_17 |
| f0v2_oft | 75 | 0.0043 | 0.0108 | layer_17 |
| f0v2_oft | 100 | 0.0074 | 0.0189 | layer_17 |
| f0v2_oft | 125 | 0.1061 | 0.2696 | layer_16 |
| f0v2_oft | 150 | 0.0113 | 0.0296 | layer_17 |
| f0v2_oft | 175 | 0.0220 | 0.0567 | layer_17 |
| f0v2_oft | 200 | 0.0029 | 0.0074 | layer_17 |
| f0v2_oft | 225 | 0.0110 | 0.0285 | layer_17 |
| f0v2_oft | 250 | 0.0106 | 0.0277 | layer_17 |
| f0v2_oft | 275 | 0.0482 | 0.1235 | layer_17 |
| f0v2_multihead | 0 | 0.0000 | 0.0000 | layer_0 |
| f0v2_multihead | 25 | 0.0337 | 0.0875 | layer_16 |
| f0v2_multihead | 50 | 0.0077 | 0.0196 | layer_17 |
| f0v2_multihead | 75 | 0.0182 | 0.0470 | layer_17 |
| f0v2_multihead | 100 | 0.0661 | 0.1703 | layer_17 |
| f0v2_multihead | 125 | 0.0233 | 0.0580 | layer_17 |
| f0v2_multihead | 150 | 0.0109 | 0.0249 | layer_17 |
| f0v2_multihead | 175 | 0.0058 | 0.0141 | layer_35 |
| f0v2_multihead | 200 | 0.0057 | 0.0164 | layer_35 |
| f0v2_multihead | 225 | 0.0173 | 0.0406 | layer_17 |
| f0v2_multihead | 250 | 0.0710 | 0.1818 | layer_16 |
| f0v2_multihead | 275 | 0.0101 | 0.0213 | layer_17 |
