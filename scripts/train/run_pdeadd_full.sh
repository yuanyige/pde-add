CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--data-diff cifar10 \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save \
--npc-train all \
--lrC 0.08 \
--lrDiff 0.001 \
--schedule cosanlr \
--ls 0.12 \
--use-gmm \
--aug-train none  \
--aug-train-diff augmix-10-10 
