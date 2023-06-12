CUDA_VISIBLE_DEVICES=2 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir test \
--npc-train all \
--lrC 0.05 \
--lrDiff 0.015 \
--schedule cosanlr \
--aug-train-inplace augmix \
--aug-train augmix-10-10 \
--atk-train none \
--atk-eval none \
--severity-eval 5


