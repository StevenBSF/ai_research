# 10月29日

- 全量微调指令实例

  - ```bash
    CUDA_VISIBLE_DEVICES=4 python train_edit.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 5e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_lr_all --pretrained
    ```

- 只调分类头指令示例

  - ```bash
    CUDA_VISIBLE_DEVICES=7 python train_head.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 5e-4 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_lr_head --pretrained
    ```

- 调prompter和head

  - ``` bash
    CUDA_VISIBLE_DEVICES=4 python train_prompt.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu_vp --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-2 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_prompt --pretrained
    ```

- VPT

  - ```bash
    CUDA_VISIBLE_DEVICES=4 python train_prompt.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu_vpt --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_prompt --pretrained
    ```

  - 

- 上午在跑的代码：
  - 全量微调lr=5e-4，lr=5e-3

- 晚上在跑的代码：

  - Gpu2：prompt lr=1e-2
  - Gpu3 ：prompt lr=1e-3
  - Gpu4 ：prompt lr=1e-4

- **你可以在使用管理员权限打开CMD后，通过执行下面的命令禁用临时IPV6地址：**

  ```
  1
  netsh interface ipv6 set privacy state=disable
  ```

  随后在`控制面板`-`网络和共享中心`-`更改适配器设置`中禁用再启用你正在使用的网卡即可。即使是WIN11，任务栏自带的搜索功能或者小娜可以轻松找到控制面板。

  > 如果想要恢复的话，管理员权限CMD中执行：
  >
  > ```
  > 1
  > netsh interface ipv6 set privacy state=enable
  > ```
  >
  > 随后重启网卡即可。



# 11月1日

- 上午在跑的代码：
  - Gpu2：lr=2e-3
  - GPU3：lr=7e-3
  - GPU4：lr=2e-2