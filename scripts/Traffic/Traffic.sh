#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=512
seed=2024

data_path=./data/traffic/traffic.csv
feature_type=M
target=OT
checkpoint_dir=./checkpoints
name=traffic

n_block=1
alpha=0.0
patch=16
batch_size=32
train_epochs=20
learning_rate=0.00008
result_path=result.csv

norm=True
layernorm=False
dropout=0.1


# pred_len = 96
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --seq_len $seq_len \
  --pred_len 96 \
  --n_block $n_block \
  --alpha $alpha \
  --patch $patch \
  --norm $norm \
  --layernorm $layernorm \
  --dropout $dropout \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --result_path $result_path \
  > ./logs/LongForecasting/${name}_${seq_len}_96.log 2>&1


# pred_len = 192
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --seq_len $seq_len \
  --pred_len 192 \
  --n_block $n_block \
  --alpha $alpha \
  --patch $patch \
  --norm $norm \
  --layernorm $layernorm \
  --dropout $dropout \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --result_path $result_path \
  > ./logs/LongForecasting/${name}_${seq_len}_192.log 2>&1


# pred_len = 336
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --seq_len $seq_len \
  --pred_len 336 \
  --n_block $n_block \
  --alpha $alpha \
  --patch $patch \
  --norm $norm \
  --layernorm $layernorm \
  --dropout $dropout \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --result_path $result_path \
  > ./logs/LongForecasting/${name}_${seq_len}_336.log 2>&1


# pred_len = 720
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --seq_len $seq_len \
  --pred_len 720 \
  --n_block $n_block \
  --alpha $alpha \
  --patch $patch \
  --norm $norm \
  --layernorm $layernorm \
  --dropout $dropout \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --result_path $result_path \
  > ./logs/LongForecasting/${name}_${seq_len}_720.log 2>&1