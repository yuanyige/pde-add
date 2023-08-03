CUDA_VISIBLE_DEVICES=4 \
python train.py \
--data tinyin200 \
--protocol standard \
--desc standard \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_tin200 \
--npc-train all \
--lrC 0.01 \
--schedule cosanlr \
--aug-train-inplace augmix-10-10 \
--aug-train none \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 1

