# Beiras RNN. A Galician RRN text generator based on Beiras text
In this notebook, we develog a Recurrent Neural Network (RNN) to create a Galician language sequence generator. In this project, we use text from the Galician politician Beiras.

This work is based in [aind2-rnn](https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb)

## Description notebook
### beiras-rnn

In this notebook, we develog a Recurrent Neural Network (RNN) to create a Galician language sequence generator. In this project, we use text from the Galician politician Beiras. 
We test diferents RNN network:
* LSTM
* GRU
* GRU + Dropout

We also test ModelCheckpoint in training.

The best network for this is case is a GRU with 3 layers.

This work is based in 
https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb

### Beiras-rnn-gan
In this notebook, we develog a GAN network with Recurrent Neural Network (RNN) to create a Galician language sequence generator. In this project, we use text from the Galician politician Beiras. 

**A GAN network does not work in this case**

This work is based on 
https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb
    
The GAN implementation is based on https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN.ipynb  

### Beiras-rnn-Timedistributed

In this notebook, we develog a  Recurrent Neural Network (RNN) with TimeDistributed to create a Galician language sequence generator. In this project, we use text from the Galician politician Beiras.

Consigue bos resultados con so 100000 carazteres, pero a base de comerse a memoria, na miña maquina aws con 100000 casi se come toda a memoria.

** Get good result only using 100000 carazters for training, but it use a lot of memory **

This work is based on https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb
###  beiras-rnn-hyperparameters
In this notebook, we develog a Recurrent Neural Network (RNN) to create a Galician language sequence generator. In this project, we use text from the Galician politician Beiras. 

In beiras-rnn/Beiras-rnn.ipynb we choose  GRU as our best network, here we try some different hyperparameters:

Move the learning rate: With 0.001 work perfect, when we reduce the net does not learn.

Change to Adam Optimizer: We get similar results than using RSMprop

Reduce batch size from 500 to 32.- This increase the time and I could not finish the learning process.

It normally takes 10 hours to train the models in a g2.2xlarge AWS machine.


## Getting Started
### Create the env
conda env create -f environment.yml
### Activate the env
source activate beiras-rnn
### Lanch notebook
jupyter notebook 
### Make predicction
python predict.py [sentence]

## Built With
* [Keras](https://keras.io/)

* [Tensorflow](https://www.tensorflow.org/)

* [Jupyter](http://jupyter.org/)

## Author

* Jose Manuel Fernandez Lorenzo - jotavaladouro-
* Based in [aind2-rnn](https://github.com/udacity/aind2-rnn/blob/master/RNN_project.ipynb)

