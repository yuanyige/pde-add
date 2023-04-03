CUDA_VISIBLE_DEVICES=4 \
python train.py \
--protocol fixdiff-1.7 \
--desc wostd \
--backbone resnet-18 \
--epoch 300 \
--save-dir exps/baselines/wostd \
--npc-train 500 \
--lrC 0.05 \
--aug-train none \
--ensemble-iter-eval 3

