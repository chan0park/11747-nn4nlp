#!/bin/bash
bss='8 16 32 64'

echo "" > res/param_bs.txt
for bs in $bss
do
echo bs: $bs >> res/param_bs.txt
python src/main.py -batch_size $bs -cuda -fix_emb -init_emb | tail -3 >> res/param_bs.txt
done
