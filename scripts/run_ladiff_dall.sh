CUDA_VISIBLE_DEVICES=4 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol ladiff-augdiff \
--desc catori \
--backbone resnet-18 \
--epoch 200 \
--save-dir exps_cifar10/dall_rn18 \
--npc-train all \
--lrC 0.05 \
--lrDiff 0.1 \
--schedule cosanlr \
--aug-train augmix \
--atk-train none \
--atk-eval none \
--severity-eval 3 

