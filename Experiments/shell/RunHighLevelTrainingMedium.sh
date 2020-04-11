#!/bin/sh

#./RunHighLevelTrainingMedium.sh 200
#cd ..
pathname="./HighlevelTraining81nodes.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_set_size $1 --batch_choice  $(($VARIABLE))
done
