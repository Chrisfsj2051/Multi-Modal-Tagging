PORT=13783 nohup bash tools/dist_train.sh configs/id11_2x.py 2 &
sleep 3
PORT=17574 nohup bash tools/dist_train.sh configs/id11_efficient_b1.py 2 &
sleep 3
PORT=24179 bash tools/dist_train.sh configs/id11_auto_color_eq.py 2
sleep 3
PORT=14204 nohup bash tools/dist_train.sh configs/id11_auto_bright_contrast.py 2 &
sleep 3
PORT=19505 bash tools/dist_train.sh configs/id11_r101.py 2
