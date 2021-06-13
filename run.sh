PORT=15364 nohup bash tools/dist_train.sh configs/text_bert_adam_mul0.001.py 2 &
sleep 3
PORT=11255 bash tools/dist_train.sh configs/text_bert_adam_mul0.01.py 2
sleep 3
PORT=12566 nohup bash tools/dist_train.sh configs/text_bert_adam_mul0.01_baselr0.01.py 2 &
sleep 3
PORT=16501 bash tools/dist_train.sh configs/text_bert_sgd_mul0.001.py 2
sleep 3
PORT=27934 nohup bash tools/dist_train.sh configs/text_bert_sgd_mul0.01.py 2 &
sleep 3
PORT=15034 bash tools/dist_train.sh configs/text_bert_sgd_mul0.01_baselr0.01.py 2
sleep 3
