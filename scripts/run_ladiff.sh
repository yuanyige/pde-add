CUDA_VISIBLE_DEVICES=3 \
python train.py \
--data cifar10 \
--seed 3407 \
--protocol ladiff-augdiff \
--desc none \
--backbone resnet-18 \
--epoch 400 \
--save-dir exps_cifar10/d500 \
--npc-train 500 \
--lrC 0.1 \
--lrDiff 0.1 \
--schedule mslr \
--aug-train augmix
