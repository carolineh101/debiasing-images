#!/bin/bash

lambdas=(0.8, 1.0, 2.0)
phis=(0.01, 0.05, 0.1, 0.25)
for lambd in "${lambdas[@]}"
do
    for phi in "${phis[@]}"
    do
        out_dir="adv-128_lambda=${lambd}_phi=${phi}_balanced"
        python src/train.py --out-dir "$out_dir" --batch-size 32 -lr 0.00001 -alr 0.0001 --hidden-size 512 --lambd "$lambd" --protected-percentage "$phi" --balance-protected
        python src/test.py --out-dir "$out_dir"
    done
done
