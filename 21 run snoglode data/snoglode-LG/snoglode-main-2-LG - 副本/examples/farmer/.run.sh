#!/bin/bash

python farmer_classic.py > farmer_classic_output.txt
bash farmer_classic.sh
python farmer_skew.py > farmer_skew_output.txt
bash farmer_skew.sh