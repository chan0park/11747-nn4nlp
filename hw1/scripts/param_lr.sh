#!/bin/bash
params='0.001 0.0001 0.00005 0.00001'

echo "" > res/param_lr.txt
for p in $params
do
echo lr: $p >> res/param_lr.txt
python src/main.py -cuda -fix_emb -init_emb -l2 $p | tail -3 >> res/param_lr.txt
done
