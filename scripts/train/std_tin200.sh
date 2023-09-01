CUDA_VISIBLE_DEVICES=4 \
python train.py \
--data tin200 \
--protocol standard \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_tin200 \
--npc-train all \
--lrC 0.02 \
--schedule cosanlr \
--aug-train none

