#!/bin/sh

#./RunParagonMedium.sh 200
#cd ..
pathname="./ParagonExp.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_set_size $1 --batch_choice  $(($VARIABLE)) --size medium
done

# python3 ParagonExp.py --batch_set_size 200 --batch_choice 0 --size medium --verbose