#!/bin/bash
params='2 3 4 5 6'

echo "" > res/param_fs.txt
for p in $params
do
echo filter size: $p >> res/param_fs.txt
python src/main.py -cuda -fix_emb -init_emb -window_dim $p | tail -3 >> res/param_fs.txt
done
