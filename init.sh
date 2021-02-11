#!/bin/bash

cd /dev/shm

# <startup scripts>
if ! [ "$ hash unzip 2>/dev/null" ] ; then  # install unzip 
	sudo apt install unzip
	echo "UNZIP INSTALLED"   # checkc
fi

if ! [ "$ hash awscli 2>/dev/null" ] ; then   # install aws 
	curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
	unzip awscliv2.zip
	sudo ./aws/install
	echo "AWS INSTALLED"   # check
fi

if ! [ "$ hash git 2>/dev/null" ] ; then   # install git 
	sudo apt install git
	echo "GIT INSTALLED"   # check
fi

# cloning livermask repo 
git clone "https://github.com/SofiaEscobarG/livermask.git"  

# mounting additional storage for LiTS data  
sudo mkfs -t xfs /dev/xvdb 
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chmod -R 777 /data

# downloading LiTS images onto new device
cd /data
aws s3 cp s3://livermask-lits/lits.zip ./    
unzip lits.zip
echo "UNZIP COMPLETE"  # check

cd /dev/shm/livermask/liverhcc/
bash bash_script.sh  # run python script 
echo "DONE"