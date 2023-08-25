CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data pacs-art \
--protocol standard \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_pacs/art \
--npc-train all \
--lrC 0.01 \
--schedule cosanlr \
--aug-train none

