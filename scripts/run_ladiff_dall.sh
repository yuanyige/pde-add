CUDA_VISIBLE_DEVICES=2 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol ladiff-oridiff \
--desc woanyaug \
--backbone resnet-18 \
--epoch 400 \
--save-dir exps_cifar10/optimal \
--npc-train all \
--lrC 0.005 \
--lrDiff 0.1 \
--schedule cosanlr \
--aug-train-inplace none \
--aug-train augmix \
--atk-train none \
--atk-eval none \
--severity-eval 5


