| bs | optimizer | lr     | image_GAP             | video_GAP | text_GAP | audio_GAP |
|--- | --------- | ------ | --------------------- | --------- | -------- | --------- |
| 32 | Adam      | 0.0001 | 0.6104 (from scratch) | 0.6711    | 0.6770   | 0.6123    |
| 32 | Adam      | 0.001  | 0.6280 (from scratch) | 0.7106    | 0.6550   | 0.5976    |
| 32 | SGD       | 0.05   | 0.6400 (from scratch) | 0.6554    | 0.6160   | 0.6005    |
| 32 | SGD       | 0.02   | 0.6493 (from scratch) | 0.6667    | 0.6560   | 0.5990    |
| 32 | SGD       | 0.01   | 0.6484 (from scratch) | 0.6813    | 0.6757   | 0.5956    |
| 2  | SGD       | 0.02   | 0.7061                | 0.6808    | 0.7176   | 0.6480    |
| 4  | SGD       | 0.02   | 0.6822                | 0.6846    | 0.7048   | 0.6706    |
| 2  | Adam      | 0.001  | 0.5740                | 0.7079    | 0.7057   | 0.6485    |
| 4  | Adam      | 0.001  | 0.6408                | 0.7072    | 0.6924   | 0.6759    |

* bs是samples_per_gpu
* image最优setting: bs=2, SGD, lr=0.02
* video最优setting: bs=4, ADAM, lr=0.001
* text最优setting: bs=2, SGD, lr=0.02
* audio最优setting: bs=4, ADAM, lr=0.001