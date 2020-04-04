#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /home/ubuntu/atc/testbed
git pull




#screen -dmS MySessionName0 &
#screen -dmS MySessionName1 &
#screen -dmS MySessionName2 &
#screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 0
#" &
#screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 10
#" &
#screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 20
#"


screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 5
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 15
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/testbed;  sh cpo_81_transfer_more.sh 25
"




screen -r MySessionName0


screen -r MySessionName0
