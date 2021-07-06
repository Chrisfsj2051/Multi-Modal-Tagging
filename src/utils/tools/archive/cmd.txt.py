source ~/.bashrc
conda init
source ~/.bashrc
bash VideoStructuring/run.sh fix
conda activate pt1.6
cp -r algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd dataset/tagging/
cp -r algo-2021/dataset/videos/test_5k_2nd dataset/videos/
cd Multi-Modal-Tagging/
git config credential.helper store
git pull

cp pretrained/L_16_imagenet1k.pth /home/tione/.cache/torch/hub/checkpoints/
bash tools/extract/run_video_frame.sh
bash tools/extract/run_video_feats.sh