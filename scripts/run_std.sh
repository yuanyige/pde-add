CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--protocol standard \
--desc standard \
--backbone wideresnet-16-4 \
--epoch 200 \
--save-dir exps_cifar10/adv \
--npc-train all \
--lrC 0.05 \
--schedule cosanlr \
--aug-train none \
--atk-train linf-pgd \
--ensemble-iter-eval 1

