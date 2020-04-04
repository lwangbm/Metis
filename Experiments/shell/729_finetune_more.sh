#!/bin/sh


pathname="CPO_clustering_27hyper_reverse_729node_finetune.py"

for VARIABLE in 0 1 2
do
python3 $pathname --batch_choice $(($VARIABLE+$1))
done

# etc.