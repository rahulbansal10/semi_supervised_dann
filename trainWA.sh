#!/usr/bin/env bash


python main.py \
    --arch resnet50 --pretrained  \
    --classes 31 \
    --bottleneck 256 \
    --gpu 0 \
    --batch-size 32 \
    --print-freq 250 --train-iter 20001 --test-iter 250 \
    --lr 0.003 \
    --name WA3-0.003-20000 \
    --dataset office --traindata webcam --valdata amazon_3 \
    --traded 1.0 --tradet 0.1

