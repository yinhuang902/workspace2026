#!/bin/bash

python bilinear.py > bilinear_output.txt
bash bilinear.sh
jupyter nbconvert --to notebook --inplace --execute bilinear.ipynb