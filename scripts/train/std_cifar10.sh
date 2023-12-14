CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--protocol standard \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save \
--npc-train all \
--lrC 0.02 \
--schedule cosanlr \
--aug-train none

