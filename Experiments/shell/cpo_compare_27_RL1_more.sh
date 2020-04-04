#!/bin/sh

ALPHA=$1
pathname="CPO_unified_RL1_1_more.py"

for VARIABLE in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
python3 $pathname --alpha $ALPHA --batch_choice $(($VARIABLE+$2))
done

# etc.