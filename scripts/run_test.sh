CUDA_VISIBLE_DEVICES=6 \
python test.py \
--ckpt_path ./save/resnet-18_pdeadd-none_ntrall_\(Csgdlr0.05cosanlr-Dadamlr0.015\)_e200_b128_aug-augmix-10-10_atk-none \
--load_ckpt model-best-wodiff \
--main_task ood \
--type c15
