CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol ladiff-augdiff \
--desc catori \
--backbone resnet-18 \
--epoch 150 \
--save-dir exps_cifar10/dall_rn18_seval5 \
--npc-train all \
--lrC 0.1 \
--lrDiff 0.1 \
--schedule piecewise \
--aug-train augmix \
--atk-train none \
--atk-eval none \
--severity-eval 5

