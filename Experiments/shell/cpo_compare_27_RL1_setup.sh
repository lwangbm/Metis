#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /home/ubuntu/atc/testbed
git pull

screen -r MySessionName0


screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
#screen -dmS MySessionName3 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_compare_27_RL1_more.sh 0.1 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_compare_27_RL1_more.sh 0.5 14
" &


screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_compare_27_RL1_more.sh 0.9 14
"
#screen -S MySessionName3 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_compare_27_RL1_more.sh 0.99 14
#"
#
#screen -dmS MySessionName4 &
#screen -S MySessionName4 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_compare_27_RL1_more.sh 1.2 14
#"