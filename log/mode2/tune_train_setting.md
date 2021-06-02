| bs | optimizer | lr     | fusion_GAP            |
|--- | --------- | ------ | --------------------- | --------- | -------- | --------- |
| 32 | Adam      | 0.0001 | 0.6104 (from scratch) |


* bs是samples_per_gpu
* image最优setting: bs=2, SGD, lr=0.02
* video最优setting: bs=2, ADAM, lr=0.001
* text最优setting: bs=2, SGD, lr=0.02
* audio最优setting: bs=2, ADAM, lr=0.001