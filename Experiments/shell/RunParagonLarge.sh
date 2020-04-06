#!/bin/sh

#./RunParagonLarge.sh 2000
cd ..
pathname="./ParagonExp.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_set_size $1 --batch_choice  $(($VARIABLE)) --size large
done

# python3 ParagonExp.py --batch_set_size 2000 --batch_choice 0 --size large --verbose