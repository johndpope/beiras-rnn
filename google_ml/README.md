# Use Google Cloud ML to train Beiras RNN

In this part, we train our model in Google Cloud ML. We follow this 
[tutorial.](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction)

## Preprocess
With preprocess.py we generate 2 files to train and test. We use 20% for the test.
The files are csv separate with ",". Each line is 101 numbers representing 
a sentence from Beiras converts to numerical using ../train/dictionaries_0.pkl.
 
## Local train
We first train in local to test the code

* Create a setup.py to define the module. [See](https://stackoverflow.com/questions/43400599/no-module-named-trainer)
* You need an __init__.py in trainer for modele work. 
* Train in local and lanch tensorboard
```sh
TRAIN_FILE=./data/beiras_train.csv
EVAL_FILE=./data/beiras_eval.csv
MODEL_DIR=./output

rm -rf $MODEL_DIR


gcloud ml-engine local train --module-name trainer.task --package-path trainer/ --job-dir $MODEL_DIR -- --train-file $TRAIN_FILE --eval-files $EVAL_FILE  --eval-steps 100 --train-steps 1000 --eval-frequency 2 --checkpoint-epochs 1

tensorboard --logdir=output --port=8080
```
* Train in local in distributed mode
```sh
TRAIN_FILE=./data/beiras_train.csv
EVAL_FILE=./data/beiras_eval.csv
MODEL_DIR=./output
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ --job-dir $MODEL_DIR --distributed -- --train-file $TRAIN_FILE --eval-files $EVAL_FILE --train-steps 1000 --eval-steps 100
tensorboard --logdir=output --port=8080
```
* Test training in local

* Install tensorflow serving and tensorflow api
```sh
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
pip install tensorflow-serving-api
```
* Predict
```sh
tensorflow_model_server --port=9000 --model_name=default --model_base_path=/home/jota/ml/beiras-rnn/export-tf

python predict-tf-serving.py  pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran
```
pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneranaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp -r data/* gs://$BUCKET_NAME/data


BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
TRAIN_FILE=gs://$BUCKET_NAME/data/beiras_train.csv
EVAL_FILE=gs://$BUCKET_NAME/data/beiras_eval.csv

JOB_NAME=beiras_rnn_single_22
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     --module-name trainer.task     --package-path trainer/   --config config.yaml  --region $REGION     --     --train-files $TRAIN_FILE     --eval-files $EVAL_FILE     --eval-frequency 2 --checkpoint-epochs 1


gcloud ml-engine jobs stream-logs $JOB_NAME
gsutil ls -r $OUTPUT_PATH
tensorboard --logdir=$OUTPUT_PATH --port=8080

Train normal : 1 epoch : 10h and not fihish


gcloud ml-engine jobs cancel beiras_rnn_single_12
gcloud ml-engine jobs list


Not use the keras version from tutorial


Download export model
EXPORT_PATH=/home/jota/ml/beiras-rnn/export-tf
mkdir $EXPORT_PATH/6
gsutil cp -r $OUTPUT_PATH/export/* $EXPORT_PATH/6



tensorflow_model_server --port=9000 --model_name=default --model_base_path=$EXPORT_PATH/

 python predict-tf-serving.py  pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran


pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran en cartón ao desenvolvemento de estado e a sua propria contra tiña unha posición de que o partir de

Execucion boa gpu e a 22
INFO	2018-02-06 17:09:57 +0100	master-replica-0		Epoch 1/20
INFO	2018-02-06 17:20:52 +0100	master-replica-0		Epoch 00001: saving model to checkpoint.01.hdf5
INFO	2018-02-06 17:20:53 +0100	master-replica-0		Epoch 2/20
INFO	2018-02-06 17:31:48 +0100	master-replica-0		Epoch 00002: saving model to checkpoint.02.hdf5
INFO	2018-02-06 17:40:01 +0100	master-replica-0		Epoch 3/20
INFO	2018-02-06 17:50:52 +0100	master-replica-0		Epoch 00003: saving model to checkpoint.03.hdf5
INFO	2018-02-06 17:50:52 +0100	master-replica-0		Epoch 4/20
INFO	2018-02-06 18:01:43 +0100	master-replica-0		Epoch 00004: saving model to checkpoint.04.hdf5
INFO	2018-02-06 18:09:56 +0100	master-replica-0		Epoch 5/20
INFO	2018-02-06 18:20:55 +0100	master-replica-0		Epoch 00005: saving model to checkpoint.05.hdf5
INFO	2018-02-06 18:20:55 +0100	master-replica-0		Epoch 6/20
INFO	2018-02-06 18:32:00 +0100	master-replica-0		Epoch 00006: saving model to checkpoint.06.hdf5
INFO	2018-02-06 18:40:24 +0100	master-replica-0		Epoch 7/20
INFO	2018-02-06 18:51:25 +0100	master-replica-0		Epoch 00007: saving model to checkpoint.07.hdf5
INFO	2018-02-06 18:51:26 +0100	master-replica-0		Epoch 8/20
INFO	2018-02-06 19:02:23 +0100	master-replica-0		Epoch 00008: saving model to checkpoint.08.hdf5
INFO	2018-02-06 19:21:40 +0100	master-replica-0		Epoch 00009: saving model to checkpoint.09.hdf5
INFO	2018-02-06 19:21:41 +0100	master-replica-0		Epoch 10/20
INFO	2018-02-06 19:32:37 +0100	master-replica-0		Epoch 00010: saving model to checkpoint.10.hdf5
INFO	2018-02-06 19:40:56 +0100	master-replica-0		Epoch 11/20
INFO	2018-02-06 19:51:54 +0100	master-replica-0		Epoch 00011: saving model to checkpoint.11.hdf5
INFO	2018-02-06 19:51:54 +0100	master-replica-0		Epoch 12/20
INFO	2018-02-06 20:02:53 +0100	master-replica-0		Epoch 00012: saving model to checkpoint.12.hdf5
INFO	2018-02-06 20:11:11 +0100	master-replica-0		Epoch 13/20
INFO	2018-02-06 20:22:12 +0100	master-replica-0		Epoch 00013: saving model to checkpoint.13.hdf5
INFO	2018-02-06 20:22:12 +0100	master-replica-0		Epoch 14/20
INFO	2018-02-06 20:33:12 +0100	master-replica-0		Epoch 00014: saving model to checkpoint.14.hdf5
INFO	2018-02-06 20:41:32 +0100	master-replica-0		Epoch 15/20
INFO	2018-02-06 20:52:32 +0100	master-replica-0		Epoch 00015: saving model to checkpoint.15.hdf5
INFO	2018-02-06 20:52:32 +0100	master-replica-0		Epoch 16/20
INFO	2018-02-06 21:03:33 +0100	master-replica-0		Epoch 00016: saving model to checkpoint.16.hdf5
INFO	2018-02-06 21:11:55 +0100	master-replica-0		Epoch 17/20
INFO	2018-02-06 21:22:52 +0100	master-replica-0		Epoch 00017: saving model to checkpoint.17.hdf5
INFO	2018-02-06 21:22:52 +0100	master-replica-0		Epoch 18/20
INFO	2018-02-06 21:33:49 +0100	master-replica-0		Epoch 00018: saving model to checkpoint.18.hdf5
INFO	2018-02-06 21:42:11 +0100	master-replica-0		Epoch 19/20
INFO	2018-02-06 21:53:08 +0100	master-replica-0		Epoch 00019: saving model to checkpoint.19.hdf5
INFO	2018-02-06 21:53:08 +0100	master-replica-0		Epoch 20/20
INFO	2018-02-06 22:04:05 +0100	master-replica-0		Epoch 00020: saving model to checkpoint.20.hdf5

Slow,it is repeating the work in all nodes 


config_distributed.yaml 
trainingInput:
  scaleTier: CUSTOM
  masterType : large_model
  workerCount : 4
  workerType : large_model
  
  
Several gpus

trainingInput:
  #scaleTier: BASIC_GPU
  scaleTier: CUSTOM
  masterType: complex_model_m_gpu
  
JOB_NAME=beiras_rnn_gpus_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     --module-name trainer.task     --package-path trainer/   --config config.yaml  --region $REGION     --     --train-files $TRAIN_FILE     --eval-files $EVAL_FILE    --gpus=4

Tutorial keras & gpu
https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/

fhttps://keras.io/utils/#multi_gpu_model
https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus

gcloud ml-engine jobs cancel  $JOB_NAME



For several nodes, make a cluster an run
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

from keras import backend as K
K.set_session(sess)

