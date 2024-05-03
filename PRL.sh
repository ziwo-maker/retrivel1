#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 nohup python -m experment.select_PRL_all-Mini &
#sleep 4h
CUDA_VISIBLE_DEVICES=0 nohup python -m experment.select_PRL_bert &
sleep 4h
CUDA_VISIBLE_DEVICES=0 nohup python -m experment.select_PRL_selector &
echo 'success'
