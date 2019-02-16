#!/bin/bash
python src/main.py -cuda -fix_emb -init_emb | tail -3 > res/final.txt
