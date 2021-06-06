* hmc (all with aug (color, flip, video_frame)) 去掉aug会掉点
| optimizer | lr     | bs | fc_GAP | hmc_GAP |
| --------- | ------ | -- | ------ | ------- |
| Adam      | 0.0001 | 2  |        | 0.7514  |
| Adam      | 0.001  | 2  |        | 0.7485  |
| Adam      | 0.001  | 8  |        | 0.7401  |
| SGD       | 0.02   | 2  |        | 0.7509  |
| SGD       | 0.05   | 2  |    0.7369    | 0.7519  |
| SGD       | 0.05   | 8  |        | 0.7505  |
| SGD       | 0.1    | 2  |        | 0.7500  |
