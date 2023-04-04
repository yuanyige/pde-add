# ladiff

### how to run and test
'''
bash ./scripts/run_ladiff_dall.sh # 需要调的参数都在下面了
'''
'''
scripts/run_test.sh # 需要更改里面的 --ckpt_path
'''

### main hyperparam to tune:

- backbone  preresnet-18  or resnet-18  or wideresnet (主实验准备在 preresnet-18  or resnet-18 上跑，谁好选谁；之后也要做 wideresnet 的补充实验)
- epoch: 400 for now 
- lrC 0.008 for now (分类器的lr，resnet18上，目前为止 lr=0.008 和 lr=0.005 比 0.01 好很多)
- lrDiff 0.1 for now (没有调过)
- schedule cosanlr for now (只是分类器的 scheduler，diffusion 部分没有设置 scheduler)
- aug-train augmix-7-6 for now (经验上来看，只要不丢失语意信息，增强越强越好，-7-6是增强强度的参数)


### Goal
- scripts/run_test.sh
  - 若 severity = 0 则平均 acc=89 是 bechmark 上 sota
  - 若 severity = 5 则平均 acc=79 是 sota

- 我目前跑完的结果只能做到：
  - severity = 0 时 acc=85.36
  - severity = 5 时 acc=77.85


### I Love U
我是你的小狗狗
