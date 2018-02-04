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

gcloud ml-engine local train --module-name trainer.task --package-path trainer/ --job-dir $MODEL_DIR -- --train-file $TRAIN_FILE --eval-files $EVAL_FILE --train-steps 1000 --eval-steps 100

tensorboard --logdir=output --port=8080
```
* Train in local in distributed mode
```sh
TRAIN_DATA=./data/beiras_train.csv
EVAL_DATA=./data/beiras_eval.csv
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
python predict-tf-serving.py  pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran
```
pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneranaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp -r data/* gs://$BUCKET_NAME/data
TRAIN_DATA=gs://$BUCKET_NAME/data/beiras_train.csv
EVAL_DATA=gs://$BUCKET_NAME/data/beiras_eval.csv
 JOB_NAME=beiras_rnn_single_5
 OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME


gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     --module-name trainer.task     --package-path trainer/     --region $REGION     --     --train-files $TRAIN_DATA     --eval-files $EVAL_DATA     --train-steps 1000     --eval-steps 100     --verbosity DEBUG


gcloud ml-engine jobs stream-logs $JOB_NAME
gsutil ls -r $OUTPUT_PATH

Train normal : 1 epoch : 10h and not fihish




