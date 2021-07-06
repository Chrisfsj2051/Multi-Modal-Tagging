source ~/.bashrc
conda init
conda create -n pt1.6 python=3.7.6 -y
conda activate pt1.6
source ~/.bashrc
conda activate pt1.6
conda install cudatoolkit==10.1.243 -y
conda install pytorch==1.6 -c pytorch -y
conda install torchvision -c pytorch -y
pip install pytest-runner -q
pip install mmcv==1.2.4 -q





