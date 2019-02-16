#!/bin/bash
params='0.01 0.001 0.0001 0.00001'

echo "" > res/param_l2.txt
for p in $params
do
echo l2: $p >> res/param_l2.txt
python src/main.py -cuda -fix_emb -init_emb -l2 $p | tail -3 >> res/param_l2.txt
done
