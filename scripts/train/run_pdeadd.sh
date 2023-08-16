CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data cifar10 \
--data-diff cifar10 \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-34 \
--epoch 200 \
--save-dir save_res34 \
--npc-train 500 \
--lrC 0.05 \
--lrDiff 0.017 \
--schedule cosanlr \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10


