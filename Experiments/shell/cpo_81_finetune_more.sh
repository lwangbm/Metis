#!/bin/sh


pathname="CPO_clustering_27hyper_reverse_81node_finetune.py"

for VARIABLE in 0 1 2 3 4
do
python3 $pathname --batch_choice $(($VARIABLE+$1))
done

# etc.