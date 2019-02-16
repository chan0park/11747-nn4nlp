#!/bin/bash
params='10 50 100 500'

echo "" > res/param_out.txt
for p in $params
do
echo out dim: $p >> res/param_out.txt
python src/main.py -cuda -fix_emb -init_emb -out_dim $p | tail -3 >> res/param_out.txt
done
