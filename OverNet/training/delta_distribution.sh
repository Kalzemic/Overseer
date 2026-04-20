#!/usr/bin/env bash


 for i in {0..9}; do 
    for j in ""; do 
        python delta_distribution.py "./data/resnet${j}/resnet${j}_train_0${i}.bin"
    done 
 done