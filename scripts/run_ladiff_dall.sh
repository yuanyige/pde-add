CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol ladiff-augdiff \
--desc none \
--backbone resnet-18 \
--epoch 400 \
--save-dir exps_cifar10/dall_rn18 \
--npc-train all \
--lrC 0.005 \
--lrDiff 0.1 \
--schedule cosanlr \
--aug-train augmix-7-6 \
--atk-train none \
--severity-eval 3 \
--ensemble-iter-eval 3

