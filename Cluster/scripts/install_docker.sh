# Docker CE install
## https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1
## verified by 20190325

sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

# To check whether it is like
##
## pub   4096R/0EBFCD88 2017-02-22
##       Key fingerprint = 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
## uid                  Docker Release (CE deb) <docker@docker.com>
## sub   4096R/F273FCD8 2017-02-22
##

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

# DONE
# To install other versions of Docker
## apt-cache madison docker-ce
## sudo apt-get install docker-ce=17.12.0~ce-0~ubuntu

# sudo docker run hello-world
sudo docker run alpine
# sudo docker run ubuntu:18.04

# docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)"  -o /usr/local/bin/docker-compose
sudo mv /usr/local/bin/docker-compose /usr/bin/docker-compose
sudo chmod +x /usr/bin/docker-compose


sudo groupadd docker
sudo usermod -aG docker $USER
logout # and login again to free from "sudo" before docker
