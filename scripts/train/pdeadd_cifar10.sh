CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--data-diff cifar10 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save \
--npc-train all \
--lrC 0.06 \
--lrDiff 0.001 \
--schedule cosanlr \
--ls 0.12 \
--use-gmm \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10 