mkdir -p pretrained/bert
cd pretrained/bert
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
tar -xzvf bert-base-chinese.tar.gz
mv bert-base-chinese-vocab.txt vocab.txt