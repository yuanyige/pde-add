CUDA_VISIBLE_DEVICES=3 \
python train.py \
--data cifar10 \
--protocol fixdiff-0.1 \
--desc rse \
--backbone resnet-18 \
--epoch 200 \
--save-dir exps_cifar10/baselines_new/ \
--npc-train all \
--lrC 0.05 \
--schedule cosanlr \
--aug-train-inplace none \
--aug-train none \
--atk-train none \
--atk-eval none \
--ensemble-iter-eval 3

