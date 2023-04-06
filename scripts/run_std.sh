CUDA_VISIBLE_DEVICES=6 \
python train.py \
--data cifar10 \
--protocol standard \
--desc standard \
--backbone resnet-18 \
--epoch 400 \
--save-dir exps_cifar10/baselines \
--npc-train all \
--lrC 0.01 \
--schedule cosanlr \
--aug-train augmix \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 1

