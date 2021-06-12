#PORT=15477 bash tools/dist_train.sh configs/id11_efficient_b1_drop0.5.py 2
#PORT=24859 bash tools/dist_train.sh configs/id11_efficient_b0_drop0.5_reverse_rgb.py 2
PORT=19782 nohup bash tools/dist_train.sh configs/id11_efficient_b0_drop0.5.py 2 &
sleep 3
PORT=20996 bash tools/dist_train.sh configs/id11_efficient_b2_drop0.5.py 2
PORT=25205 nohup bash tools/dist_train.sh configs/id11_efficient_b3_drop0.5.py 2 &
sleep 3
PORT=15021 bash tools/dist_train.sh configs/id11_efficient_b4_drop0.5.py 2
