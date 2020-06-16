#!/bin/bash

# test bash script

dbfile="./trainingdata.csv"
rootloc="/data/LiTS"
numepochs=10
outdir="./AWS_test"
kfolds=1
predictimage="/data/LiTS/TrainingBatch1/volume-6.nii"

echo "python3 tumorhcc.py --dbfile=$dbfile --builddb --rootlocation=$rootloc" > tumor_output.out
echo "python3 tumorhcc.py --dbfile=$dbfile --builddb --rootlocation=$rootloc" > tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --builddb --rootlocation=$rootloc >> tumor_output.out 2>> tumor_output.err

echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --idfold=0 --trainmodel --outdir=$outdir" >> tumor_output.out
echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --idfold=0 --trainmodel --outdir=$outdir" >> tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --idfold=0 --trainmodel --outdir=$outdir >> tumor_output.out 2>> tumor_error.err

echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --outdir=$outdir" >> tumor_output.out
echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --outdir=$outdir" >> tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --outdir=$outdir >> tumor_output.out 2>> tumor_error.err

echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$outdir" >> tumor_output.out
echo "python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$outdir" >> tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$outdir >> tumor_output.out 2>> tumor_error.err
