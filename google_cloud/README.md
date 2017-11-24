# Deploy model in Google Cloud ML, Create an API and a Web for Beiras RNN

First ee deploy our model to Google Cloud ML, and work with in python.

Then we develop a REST API to deploy our RNN Model, we use Google App Engine to server this API and create a web page to test it.

We use python2.7 for this part, beucase it is used bu Google Cloud.

## Instalation

### Intall gloud and libs
```sh
apt-get install curl
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
# Create an environment variable for the correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

apt-get install google-cloud-sdk-app-engine-python
gcloud init
gcloud auth application-default login
gcloud config set project ai-ml-dl
```
### Create python env and install libs
```sh
conda env create -f  google-cloud-ml.yml
source activate google-cloud-ml
pip install webtest webapp2 grpcio pip install grpc
pip install requests
pip install --upgrade google-api-python-client
```


## Deploy to Google Cloud ML

### Test Tensorflow model in local


```sh
gcloud ml-engine local predict --model-dir=/home/jota/beiras-rnn/export-tf/3 --json-instances=/home/jota/beiras-rnn/predict/data-ml-local.json > predict/return.json
```

### Deploy model

Make bucket and create model
```sh
MODEL_NAME=BeirasRnn
gcloud ml-engine models create $MODEL_NAME --enable-logging
gsutil mb gs://beiras_rnn
```

Upload model

```sh
gsutil cp -r ./export-tf/3 gs://beiras_rnn/export-google-ml/10
MODEL_NAME=BeirasRnn
DEPLOYMENT_SOURCE="gs://beiras_rnn/export-google-ml/10"
gcloud ml-engine versions create "v10" --model $MODEL_NAME  --origin $DEPLOYMENT_SOURCE
```


### Get sequence from Google Cloud ML

Use *Google ML Python 2.7* notebook as example

```sh
curl -i -X POST https://ml.googleapis.com/v1/projects/ai-ml-dl/models/BeirasRnn/versions/v10:predict -H "Authorization: Bearer `gcloud auth print-access-token`" -H "Content-Type: application/json" -d @data-ml-cloud.json
```

```sh
source activate google-cloud-ml
cd predict
python predict-google-cloud-ml.py se moito cando dixen eu que as suas políticas agresoras do común cidadán matan e a sua cospedal alcu
```


## Google App engine API


### Lib needed in app engine
The python library that you need to use in the App Engine  must be in lib dir

```sh
source activate google-cloud-ml
cd predict
pip install -t lib google-api-python-client grpcio
```
### Test app local

```sh
source activate google-cloud-ml
dev_appserver.py app.yaml --log_level=debug --host=localhost
curl http://localhost:8080/api/beiras_rnn -H "Content-Type: application/json" -X POST -d '{"input" : "Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o marco"}'
```


### Deply app
```sh
gcloud app deploy -v 11
curl http://ai-ml-dl.appspot.com/api/beiras_rnn -H "Content-Type: application/json" -X POST -d '{"input" : "Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o marco"}'
```

## Python script
```sh

../predict/predict-app-engine.py Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o
```

### Web test
[beiras Rnn page](http://ai-ml-dl.appspot.com/index.html)

### Api description
* URL : http://localhost:8080/api/beiras_rnn
* Method: POST
* Data Params
    * Required : input=[string]
* Success Response:
    * Code: 200
    * Content: output=[string]
* Error response:
    * Code 400: Content: { error_code : 1} .- Wrong input text
    * Code 400: Content: { error_code : 2} .- Input text lenght<100
    * Code 500: Content: { error_code : 1} .- Internal error


