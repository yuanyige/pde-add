CUDA_VISIBLE_DEVICES=4 \
python train.py \
--data tin200 \
--data-diff tin200 \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_tin200 \
--npc-train all \
--lrC 0.02 \
--lrDiff 0.001 \
--schedule cosanlr \
--ls 0.1 \
--use-gmm \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10 
