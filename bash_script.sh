#!/bin/bash

cd /dev/shm/livermask/liverhcc/

# activate tensorflow and install missing package
source activate tensorflow_p36              
conda install -c conda-forge nibabel

# installing zip 
sudo apt install zip

#running liverhcc code
dbfile="./trainingdata.csv"
rootloc="/data/LiTS" #mounted device
numepochs=20
outdir="./shuffle_v1-1"
kfolds=1
thickness=3
gpu=1
trbatch=10  
valbatch=10 
predictmodel="./shuffle_v1-1/001/000/tumor/modelunet.h5"
predictimage="/data/LiTS/TrainingBatch2/volume-30.nii"
segment="/data/LiTS/TrainingBatch2/segmentation-30.nii"

echo "BEGIN DATABASE BUILDING"
nohup python3 tumorhcc.py --dbfile=$dbfile --builddb --rootlocation=$rootloc 
echo "BUILDDB COMPLETE"


echo "python3 tumorhcc.py --numepochs=$numepochs --kfolds=$kfolds --trainmodel --thickness=$thickness --D3" > tumor_output.out
echo "python3 tumorhcc.py --numepochs=$numepochs --kfolds=$kfolds --trainmodel --thickness=$thickness --D3" > tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --numepochs=$numepochs --kfolds=$kfolds --trainmodel --outdir=$outdir --thickness=$thickness --D3 --gpu=$gpu --trainingbatch=$trbatch --validationbatch=$valbatch >> tumor_output.out 2>> tumor_error.err

echo "TRAINING COMPLETE"

echo "python3 tumorhcc.py --predictmodel=$predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$segment" >> tumor_output.out
echo "python3 tumorhcc.py --predictmodel=$predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$segment" >> tumor_error.err
nohup python3 tumorhcc.py --dbfile=$dbfile --rootlocation=$rootloc --predictmodel=$predictmodel --outdir=$outdir --predictimage=$predictimage --segmentation=$segment >> tumor_output.out 2>> tumor_error.err

echo "PREDICTION COMPLETE"

mv tumor_output.out ./shuffle_v1-1/tumor_output.out
mv tumor_error.err ./shuffle_v1-1/tumor_error.err
zip -r shuffle_v1-1.zip ./shuffle_v1-1