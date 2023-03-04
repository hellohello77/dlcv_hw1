#!/bin/bash

# TODO - run your inference Python3 code

start=$(date +%s)
python3 hw1_2_inf.py $1 $2
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
