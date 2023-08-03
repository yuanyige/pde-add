CUDA_VISIBLE_DEVICES=1 \
python test_in.py \
--ckpt_path /home/yuanyige/Ladiff_nll/save_tin200/resnet-18_standard-standard_ntrall_Csgdlr0.01cosanlr_e200_b128_aug-none_atk-none \
--load_ckpt model-best \
--main_task ood \
--type c15
