CUDA_VISIBLE_DEVICES=5 \
python train.py \
--data cifar100 \
--protocol standard \
--desc gaublur \
--backbone resnet-18 \
--epoch 200 \
--save-dir exps_cifar100/baselines \
--npc-train all \
--lrC 0.05 \
--schedule cosanlr \
--aug-train-inplace gaublur \
--aug-train none \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 1

