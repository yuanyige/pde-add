CUDA_VISIBLE_DEVICES=2 \
python test.py \
--ckpt_path exps_cifar10/dall_rn18/resnet-18_ladiff-augdiff-none_ntrall_\(Csgdlr0.05cosanlr-Dadamlr0.1\)_e200_b128_aug-augmix-10-10_atk-none \
--load_ckpt model-best-endiff \
--main_task ood