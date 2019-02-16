#!/bin/bash
# echo "Full\n" >> res/ablation.txt
# python src/main.py -batch_size 16 -cuda -epochs 10 -fix_emb | tail -3 >> res/final.txt
echo ablation > res/ablation.txt
echo "-Glove\n" >> res/ablation.txt
python src/main.py -cuda | tail -3 >> res/ablation.txt
echo "-Fix-emb\n" >> res/ablation.txt
python src/main.py -cuda -init_emb | tail -3 >> res/ablation.txt
echo "-l2\n" >> res/ablation.txt
python src/main.py -cuda -init_emb -fix_emb -l2 0.0 | tail -3 >> res/ablation.txt
echo "-dropout\n" >> res/ablation.txt
python src/main.py -cuda -fix_emb -init_emb -dp 0.0 | tail -3 >> res/ablation.txt
