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
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 10
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 20
"










screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 3
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 13
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 23
"












screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 6
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 7
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh 729_transfer_more.sh 8
"

screen -r MySessionName0

