#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /home/ubuntu/atc/testbed
git pull




screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  python3 CPO_clustering_27hyper_reverse_729node_pre_vio_4.py --batch_choice 1000
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  python3 CPO_clustering_27hyper_reverse_729node_pre_vio_4.py --batch_choice 2000
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  python3 CPO_clustering_27hyper_reverse_729node_pre_vio_4.py --batch_choice 3000
"
screen -r MySessionName0


screen -dmS MySessionName4
screen -S MySessionName4 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  python3 CPO_clustering_27hyper_reverse_729node_pre_vio_4.py --batch_choice 2000
"