# Beiras RNN. A Galician RRN text generator based on Beiras text
In this project, we develop a Recurrent Neural Network (RNN) to create a Galician language sequence generator. We use text from the Galician politician Beiras and keras for defining and training.

This work is based in [aind2-rnn](https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb)

First, we train different RNN, using different techniques and hyperparameter and we choose the best: 2 layer GRU + 1 Dense.

Then we generate secuences:
* In local, defining the model and loading the weights.
* In remote, saving the model as a tensorflow model and using *tensorflow-serving*.



## Notebook for train (./train)
### beiras-rnn

We develog a Recurrent Neural Network (RNN) to create a Galician language sequence generator.

We test diferents RNN network:
* LSTM
* GRU
* GRU + Dropout

We also test ModelCheckpoint in training.

The best network for this is case is a GRU with 3 layers.

This work is based in [aind2-rnn](https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb)

### Beiras-rnn-gan
We develop a GAN network with Recurrent Neural Network (RNN) to create a Galician language sequence generator. 


**A GAN network does not work in this case**
    
The GAN implementation is based on [MNIST_CNN_GAN](https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN.ipynb)  

### Beiras-rnn-Timedistributed

We develog a  Recurrent Neural Network (RNN) with TimeDistributed to create a Galician language sequence generator.

** Get good result only using 100000 carazters for training, but it use a lot of memory **


###  beiras-rnn-hyperparameters

In beiras-rnn/Beiras-rnn.ipynb we choose  GRU as our best network, here we try some different hyperparameters:

Move the learning rate: With 0.001 work perfect, when we reduce the net does not learn.

Change to Adam Optimizer: We get similar results than using RSMprop

Reduce batch size from 500 to 32.- This increase the time and I could not finish the learning process.

It normally takes 10 hours to train the models in a g2.2xlarge AWS machine.

## Predict (./train)

### predict-local.py
Create the model using the weights and make a sequence from the sentence introduce in the command line.

### predict-server.py
Make a secuence from the sentence introduce in the command line using tensorflow-servin.


## Getting Started
### Create the env
```sh
conda env create -f environment.yml
```
### Activate the env
```sh
source activate beiras-rnn
```
### Install tensorflow-model-server
```sh
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```

### Lanch notebook
```sh
jupyter notebook 
```
### Make predicction local
```sh
cd predict
python predict-local.py [sentence]
```
### Make predicction using tensorflow-server
```sh
tensorflow_model_server --port=9000 --model_name=default --model_base_path=/home/aind2/beiras-rnn/export-tf &
cd predict
python predict-remote.py [sentence]
```
## Built With
* [Keras](https://keras.io/)

* [Tensorflow](https://www.tensorflow.org/)

* [Jupyter](http://jupyter.org/)

## Author

* Jose Manuel Fernandez Lorenzo - jotavaladouro-
* Based in [aind2-rnn](https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb)

