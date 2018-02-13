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


gcloud ml-engine local train --module-name trainer.task --package-path trainer/ --job-dir $MODEL_DIR -- 
    --train-file $TRAIN_FILE --eval-files $EVAL_FILE  --eval-steps 100 --train-steps 1000 --eval-frequency 2 --checkpoint-epochs 1

tensorboard --logdir=output --port=8080
```
* Train in local in distributed mode
```sh
TRAIN_FILE=./data/beiras_train.csv
EVAL_FILE=./data/beiras_eval.csv
MODEL_DIR=./output
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ --job-dir $MODEL_DIR --distributed --
 --train-file $TRAIN_FILE --eval-files $EVAL_FILE --train-steps 1000 --eval-steps 100
tensorboard --logdir=output --port=8080
```
* Test training in local

 Install tensorflow serving and tensorflow api
```sh
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" 
| sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
pip install tensorflow-serving-api
```
 Predict
```sh
tensorflow_model_server --port=9000 --model_name=default --model_base_path=/home/jota/ml/beiras-rnn/export-tf

python predict-tf-serving.py  pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran
```
pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneranaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

## Cloud train
* Create the bucket and upload training tada
```sh
BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp -r data/* gs://$BUCKET_NAME/data
```


* Lanch the training in a basic node. This is too slow. From the tutorial you need to change the setup.py
 in order to use the default keras version.
```sh
BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
TRAIN_FILE=gs://$BUCKET_NAME/data/beiras_train.csv
EVAL_FILE=gs://$BUCKET_NAME/data/beiras_eval.csv
JOB_NAME=beiras_rnn_single_30
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     
    --module-name trainer.task  --package-path trainer/  --scale-tier BASIC   --region $REGION     --
    --train-files $TRAIN_FILE     --eval-files $EVAL_FILE    
```


* Lanch the training in a node with a gpu. It need 10 minutes to train a epoch.
```sh
BUCKET_NAME="beiras_rnn_mlengine"
REGION=us-central1
TRAIN_FILE=gs://$BUCKET_NAME/data/beiras_train.csv
EVAL_FILE=gs://$BUCKET_NAME/data/beiras_eval.csv
JOB_NAME=beiras_rnn_single_30
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     
    --module-name trainer.task     --package-path trainer/  --scale-tier BASIC_GPU   --region $REGION   
     --     --train-files $TRAIN_FILE     --eval-files $EVAL_FILE    
```


### Commands for cloud train

See the log
```sh
gcloud ml-engine jobs stream-logs $JOB_NAME
```
See the output
```sh
gsutil ls -r $OUTPUT_PATH
```
Use tensorboard to see the output
```sh
tensorboard --logdir=$OUTPUT_PATH --port=8080
```
Get jobs list
```sh
gcloud ml-engine jobs cancel beiras_rnn_single_12
```
Cancel a job
```sh
gcloud ml-engine jobs list
```


### Download and export the model
```sh
EXPORT_PATH=/home/jota/ml/beiras-rnn/export-tf
mkdir $EXPORT_PATH/6
gsutil cp -r $OUTPUT_PATH/export/* $EXPORT_PATH/6
python predict-tf-serving.py  pla panfletaria contra as leoninas taxas impostas polo ministro de xustiza actual malia que vulneran
```


## Cloud train distributed


```sh
 gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4 
     --module-name trainer.task     --package-path trainer/   --scale-tier STANDARD_1  --region $REGION    
     --   --train-files $TRAIN_FILE     --eval-files $EVAL_FILE 
 ```

I tried this, but nodes go out of memory and the training is repeating in all nodes, not distributing it.

I also try with this config file config_distributed.yaml
```yaml 
trainingInput:
  scaleTier: CUSTOM
  masterType : large_model
  workerCount : 4
  workerType : large_model
```


## Cloud train several gpus

I try this config file
```yaml 
trainingInput:
  #scaleTier: BASIC_GPU
  scaleTier: CUSTOM
  masterType: complex_model_m_gpu
```
  

```sh
JOB_NAME=beiras_rnn_gpus_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     
    --module-name trainer.task     --package-path trainer/   --config config.yaml  --region $REGION     
    --     --train-files $TRAIN_FILE     --eval-files $EVAL_FILE    --gpus=4
```

But it was slower than with only one GPU


[Tutorial keras gpu 1](
https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/)
[Tutorial keras gpu 2](
https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus)
[Keras multi gpu podel](https://keras.io/utils/#multi_gpu_model)

[Discussion about why keras is slow in multi gpu](https://github.com/keras-team/keras/issues/9204)
### Change in code to use sevarals gpus
With several GPU you use 2 models, one for training and other for store.
The first one is assigned to the CPU, the other run in the GPU and is generated using multi_gpu_model
```python
  if gpus <= 1:
    model_train = model.model_fn(NUM_CHARS,window_size=WINDOWS_SIZE)
    model_save = model_train
  else:
    with tf.device("/cpu:0"):
      model_save = model.model_fn(NUM_CHARS, window_size=WINDOWS_SIZE)
    model_train = multi_gpu_model(model_save, gpus=gpus)
    model.compile_model(model_save, learning_rate)
    print(model_save.summary())
  model.compile_model(model_train, learning_rate)
  print(model_train.summary())
```





