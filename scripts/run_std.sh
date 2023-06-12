CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data cifar100 \
--protocol standard \
--desc standard \
--backbone resnet-18 \
--epoch 200 \
--save-dir save \
--npc-train all \
--lrC 0.05 \
--schedule cosanlr \
--aug-train-inplace none \
--aug-train none \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 1

