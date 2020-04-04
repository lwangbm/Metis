#!/bin/sh

#./RunFPO.sh 0.1 0
ALPHA=$1
pathname="FPO.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
python3 $pathname --alpha $ALPHA --batch_choice $(($VARIABLE+$2))
done
