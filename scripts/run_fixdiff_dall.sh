CUDA_VISIBLE_DEVICES=1 \
python train.py \
--protocol fixdiff-0.5 \
--desc rse-wostd \
--backbone resnet-18 \
--epoch 300 \
--save-dir exps_cifar10/baselines/wostd \
--npc-train all \
--lrC 0.05 \
--aug-train none \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 3

