CUDA_VISIBLE_DEVICES=6 \
python test.py \
--ckpt_path exps_cifar100/dall_rn18/resnet-18_ladiff-augdiff-ls0.1_ntrall_\(Csgdlr0.05cosanlr-Dadamlr0.017\)_e200_b128_aug-augmix_atk-none \
--load_ckpt model-best-wodiff \
--main_task ood \
--type c15
