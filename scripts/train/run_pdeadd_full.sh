CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data cifar10 \
--data-diff cifar10 \
--seed 3407 \
--protocol pdeadd \
--desc none \
--backbone resnet-18 \
--epoch 200 \
--save-dir save_cifar10_new \
--npc-train all \
--lrC 0.05 \
--lrDiff 0.017 \
--schedule cosanlr \
--aug-train augmix-10-10 \
--aug-train-diff augmix-10-10 
