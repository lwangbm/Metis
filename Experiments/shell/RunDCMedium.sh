#!/bin/sh

#./RunDCMedium.sh 200
#cd ..
pathname="./DC81nodes.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_set_size $1 --batch_choice  $(($VARIABLE))
done
