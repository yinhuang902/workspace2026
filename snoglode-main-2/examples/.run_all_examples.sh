#!/bin/bash

dirs=(bilinear farmer ip pmedian quad knapsack)
for dir in "${dirs[@]}"; do
    cd $dir
    bash .run.sh
    cd ..
done