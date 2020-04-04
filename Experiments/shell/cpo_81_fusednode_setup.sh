#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /home/ubuntu/atc/testbed
git pull

sudo rm -r checkpoint
mkdir checkpoint


screen -dmS 80MySessionName0 &
screen -dmS 80MySessionName1 &
screen -dmS 80MySessionName2 &
screen -S 80MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_usednode_more.sh 0
" &
screen -S 80MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_usednode_more.sh 10
" &
screen -S 80MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_usednode_more.sh 20
"
screen -r 80MySessionName0

