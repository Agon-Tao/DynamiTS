#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=512
seed=2024

data_path=./data/ETTm2.csv
feature_type=M
target=OT
checkpoint_dir=./checkpoints
name=ETTm2

n_block=1
alpha=0.0
batch_size=128
result_path=result.csv

norm=True
layernorm=True


# pred_len = 96
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --name $name \
  --seq_len $seq_len \
  --pred_len 96 \
  --n_block $n_block \
  --alpha $alpha \
  --patch 4 \
  --norm $norm \
  --layernorm $layernorm \
  --dropout 0.1 \
  --train_epochs 10 \
  --batch_size $batch_size \
  --learning_rate 0.00003 \
  --result_path $result_path \
  --use_adaptive_fredf \
  --auxi_mode fft \
  --rec_lambda 0.8 \
  --auxi_lambda 0.2 \
  --auxi_type complex-mag-phase \
  --auxi_loss MAE \
  > ./logs/LongForecasting/${name}_${seq_len}_96.log 2>&1


# pred_len = 192
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --name $name \
  --seq_len $seq_len \
  --pred_len 192 \
  --n_block $n_block \
  --alpha $alpha \
  --patch 4 \
  --norm $norm \
  --layernorm $layernorm \
  --dropout 0.1 \
  --train_epochs 10 \
  --batch_size $batch_size \
  --learning_rate 0.00005 \
  --result_path $result_path \
  --use_adaptive_fredf \
  --auxi_mode rfft \
  --rec_lambda 0.8 \
  --auxi_lambda 0.2 \
  --auxi_type complex-mag-phase \
  --auxi_loss MAE \
  > ./logs/LongForecasting/${name}_${seq_len}_192.log 2>&1


# pred_len = 336
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --name $name \
  --seq_len $seq_len \
  --pred_len 336 \
  --n_block $n_block \
  --alpha $alpha \
  --patch 4 \
  --norm $norm \
  --layernorm $layernorm \
  --dropout 0.3 \
  --train_epochs 10 \
  --batch_size $batch_size \
  --learning_rate 0.00005 \
  --result_path $result_path \
  > ./logs/LongForecasting/${name}_${seq_len}_336.log 2>&1


# pred_len = 720
python -u main.py \
  --seed $seed \
  --data $data_path \
  --feature_type $feature_type \
  --target $target \
  --checkpoint_dir $checkpoint_dir \
  --name $name \
  --seq_len $seq_len \
  --pred_len 720 \
  --n_block $n_block \
  --alpha $alpha \
  --patch 4 \
  --norm $norm \
  --layernorm $layernorm \
  --dropout 0.2 \
  --train_epochs 10 \
  --batch_size $batch_size \
  --learning_rate 0.00003 \
  --result_path $result_path \
  --use_adaptive_fredf \
  --auxi_mode fft \
  --auxi_type complex-mag-phase \
  --auxi_loss MAE \
  > ./logs/LongForecasting/${name}_${seq_len}_720.log 2>&1