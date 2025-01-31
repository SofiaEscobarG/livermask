#!/bin/bash

# Script for segmentation
# Jonas Actor
# 30 Jan 2019
#
# Usage:
# ./makemodel-levelset.sh <location/of/dbfile.csv> <numepochs> <kfolds> <location/of/directory/for/output> <nt>
#

dbfile=$1
numepochs=$2
kfolds=$3
outdir=$4
nt=$5
let "lastfold = $kfolds - 1"

nohup python3 liver4.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=1 --idfold=0 --trainmodel --outdir=$outdir --nt=$nt
for n in $(seq 0 $lastfold)
	do
		nohup python3 liver4.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=$kfolds --idfold=$n --trainmodel --outdir=$outdir --nt=$nt
       	done
nohup python3 liver4.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=$kfolds --setuptestset --outdir=$outdir --nt=$nt
nohup make -f kfold005-predict.makefile
mv kfold005-predict.makefile $outdir
