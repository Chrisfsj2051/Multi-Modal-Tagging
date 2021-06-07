[comment]: <> (* hmc &#40;all with aug &#40;color, flip, video_frame&#41;&#41; 去掉aug会掉点)

| optimizer| lr     | bs | fc_GAP | hmc_GAP |
|-------- |------- |--- | ------ | ------- |
| Adam      | 0.0001 | 2  |        | 0.7514  |
| Adam      | 0.001  | 2  |        | 0.7485  |
| Adam      | 0.001  | 8  |        | 0.7401  |
| SGD       | 0.02   | 2  |        | 0.7509  |
| SGD       | 0.05   | 2  |    0.7369    | 0.7519  |
| SGD       | 0.05   | 8  |        | 0.7505  |
| SGD       | 0.1    | 2  |        | 0.7500  |

done

| note              | GAP    |
|------------------ | ------ |
| baseline          |        |
| hmc feat dim=256  | 0.7494 |
| hmc feat dim=1024 | 0.7490 |
| attn drop=0.2     | 0.7494 |
| drop modal=0.3    | 0.7427 |


---


text0.7178_audio0.6702_video0.7139_image0.7039.pth

fc_baseline: 0.7479
HMC_baseline: 0.7463
fc+drop_modal0.3+attn_drop0.3: 0.72
video+text, fc: 0.7378
fc+no frame aug: 0.7396
