CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data cifar10 \
--protocol standard \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_cifar10 \
--npc-train all \
--lrC 0.01 \
--schedule cosanlr \
--aug-train none 

