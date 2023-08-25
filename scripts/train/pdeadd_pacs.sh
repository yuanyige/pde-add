CUDA_VISIBLE_DEVICES=1 \
python train.py \
--data pacs-art \
--data-diff pacs-art \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_pacs/art \
--npc-train all \
--lrC 0.02 \
--lrDiff 0.005 \
--schedule cosanlr \
--ls 0.1 \
--use-gmm \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10
