CUDA_VISIBLE_DEVICES=2 \
python train.py \
--data tinyin200 \
--seed 3407 \
--protocol pdeadd \
--desc norm-inplace \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_tin200 \
--npc-train all \
--lrC 0.05 \
--lrDiff 0.017 \
--schedule cosanlr \
--aug-train-inplace augmix-10-10 \
--aug-train augmix-10-10 \
--atk-train none \
--atk-eval none \
--severity-eval 5


