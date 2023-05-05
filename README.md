# ladiff

---
### 如何训练与测试
```
bash ./scripts/run_ladiff_dall.sh    
```

需要调的参数都在下面了，正常来说 train 完了会自动 test

如需手动指定 checkpoint 进行测试，代码如下

```
bash ./scripts/run_test.sh    # 需要更改里面的 --ckpt_path
```

