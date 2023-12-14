# PDE+: Enhancing Generalization via PDE with Adaptive Distributional Diffusion

> Yige Yuan, Bingbing Xu, Bo Lin, Liang Hou, Fei Sun, Huawei Shen, Xueqi Cheng
>
> The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024

This is a PyTorch implementation of [PDE+: Enhancing Generalization via PDE with Adaptive Distributional Diffusion](https://arxiv.org/pdf/2305.15835.pdf) (PDE+).

![PDE+](pic/model.png)


## Training & Testing

All arguments are located in the parse.py file. You can create a script to specify the parameters, for example,
```
bash ./scripts/train/pdeadd_cifar10.sh # run our PDE+
bash ./scripts/train/std_cifar10.sh # run baseline
```

## Reference

If you find our work useful, please consider citing our paper:
```
@article{yuan2023pde+,
  title={PDE+: Enhancing Generalization via PDE with Adaptive Distributional Diffusion},
  author={Yuan, Yige and Xu, Bingbing and Lin, Bo and Hou, Liang and Sun, Fei and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2305.15835},
  year={2023}
}
```