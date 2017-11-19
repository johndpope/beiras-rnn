# Create an API and a Web for Beiras RNN
We develop a REST API to deploy our RNN Model, we use Google App Engine to server this API and create a web page to test it.

## Instalation

```sh
apt-get install google-cloud-sdk-app-engine-python
source activate google-cloud-ml
pip install webtest webapp2 grpcio
cd predict
pip install -t lib google-api-python-client grpcio
```
## Test app local

```sh
source activate google-cloud-ml
dev_appserver.py app.yaml --log_level=debug --host=localhost
curl http://localhost:8080/api/beiras_rnn -H "Content-Type: application/json" -X POST -d '{"input" : "Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o marco"}'
```


## Deply app
```sh
gcloud app deploy -v 11
curl http://ai-ml-dl.appspot.com/api/beiras_rnn -H "Content-Type: application/json" -X POST -d '{"input" : "Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o marco"}'
```

## Python call
```sh
pip install requests
../predict/predict-app-engine.py Non é facil de entender o que esta pasando con cataluña, unha volta atrais ou e unha elaborada estratexia para superar o
```

## Web test
[beiras Rnn page](http://ai-ml-dl.appspot.com/index.html)
