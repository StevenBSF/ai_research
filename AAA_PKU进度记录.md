# 10月29日

- 全量微调指令实例

  - ```bash
    CUDA_VISIBLE_DEVICES=4 python train_edit.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 5e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_lr_all --pretrained
    ```

- 只调分类头指令示例

  - ```bash
    CUDA_VISIBLE_DEVICES=7 python train_head.py /data/dataset/baoshifeng/cifar100 --model vig_b_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.999 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 5e-4 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /data/ckpt/baoshifeng/vig_pytorch/saved_models_lr_head --pretrained
    ```

    

- 上午在跑的代码：
  - 全量微调lr=5e-4，lr=5e-3