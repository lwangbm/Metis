#!/bin/sh


pathname="CPO_27_nodeused_more.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
python3 $pathname --batch_choice $(($VARIABLE+$1))
done

# etc.