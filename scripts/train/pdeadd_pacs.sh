CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data pacs-art \
--data-diff pacs-cartoon \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_pacs/art-cartoon \
--npc-train all \
--lrC 0.015 \
--lrDiff 0.01 \
--schedule cosanlr \
--ls 0.1 \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10
