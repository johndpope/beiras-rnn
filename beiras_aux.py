"""
Auxilar funcions for beiras_rnn
There is 3 group of function
- Clean and load the text file
- Create a text sequeence.
- Save and load dictionaries
- A main part to test dictionaries part
"""


import numpy as np
import re
import pickle


"""
Funtions Clean and load the text file
"""

def window_transform_text(text,window_size,step_size):
    """
    Use sliding window to extract input/output pairs from a text
    params:
        text .- text to windownize
        windows_size .- Len of any input sentence to return
    Return:
        input .- array of sentence 
        output .- for any sentence in input, the next charazter
    """
    # containers for input/output pairs
    inputs = []
    outputs = []
    #Number of windows to create
    n_windows=int((len(text) - window_size)/ step_size)
    for j in range(n_windows) :
        # k .- Start index
        k= j * step_size
        inputs.append(text[k:(k+window_size)])
        outputs.append(text[k+window_size])
    return inputs,outputs


def encode_io_pairs(text,window_size,step_size,chars_to_indices):
    """
     transform character-based input/output into equivalent numerical versions
    """
    # number of unique chars
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    
    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)
    
    # loop over inputs/outputs and tranform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices[char]] = 1
        y[i, chars_to_indices[outputs[i]]] = 1
        
    return X,y


def clean_text(text_org):
    """
    Clean a text
    """
    text_without_source="";
    regexp=re.compile(r'http')
    for line in text_org.splitlines():
        if not regexp.search(line):
            text_without_source= text_without_source + line
    text_clean = re.sub('[ºªàâäçèêïìôöü&%@•…«»”“*/!"(),.:;_¿¡¿‘’´\[\]\']',' ',text_without_source)
    text_clean = text_clean.replace("  "," ") 
    return text_clean


def LoadText(sz_file,window_size,step_size):
    """
    Load a text, clean it and windownize
    """
    text_org = open(sz_file, encoding="utf-8").read().lower()
    text_clean=clean_text(text_org);
    chars=sorted(list(set(text_clean )))
    # this dictionary is a function mapping each unique character to a unique integer
    chars_to_indices = dict((c, i) for i, c in enumerate(chars))  
    # this dictionary is a function mapping each unique integer back to a unique character
    indices_to_chars = dict((i, c) for i, c in enumerate(chars))  
    X,y = encode_io_pairs(text_clean,window_size,step_size,chars_to_indices)
    return X,y,chars,chars_to_indices,indices_to_chars,text_clean


"""
Functions to create a text sequence.
"""



def predict_next_chars(model,input_chars,window_size,chars_to_indices,indices_to_chars):
    """
    Predict next window_size character from a sentence using a model
    chars_to_indices and indices to chars are dictionaries uses to translate char in index.    Must be the same that used in training the model.   
    """
    # create output
    number_chars=len(chars_to_indices)
    predicted_chars = ''
    for i in range(window_size):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, window_size, number_chars))
        for t, char in enumerate(input_chars):
            x_test[0, t, chars_to_indices[char]] = 1.

        # make this round's prediction
        test_predict = model.predict(x_test,verbose = 0)[0]

        # translate numerical prediction back to characters
        r = np.argmax(test_predict)                           # predict class of each test input
        d = indices_to_chars[r] 

        # update predicted_chars and input
        predicted_chars+=d
        input_chars+=d
        input_chars = input_chars[1:]
    return predicted_chars



def  print_predicctions(model,weights_file,chars_to_indices,indices_to_chars,text_clean,window_size):
    """
      Print predicctions for sentence beginning in position 100,1000 and 5000 of text_clean
    chars_to_indices and indices to chars are dictionaries uses to translate char in index. Must be the same that used in training the model.
    """
    start_inds = [100,1000,5000]

    # load in weights
    model.load_weights(weights_file)
    for s in start_inds:
        start_index = s
        input_chars = text_clean[start_index: start_index + window_size]

        # use the prediction function
        predict_input = predict_next_chars(model,input_chars,window_size,chars_to_indices,indices_to_chars)

        print (input_chars + "...." +  predict_input)
        
        
"""
Functions to  save and load dictionaries        
"""

def save_coded_dictionaries(chars_to_indices,indices_to_chars):
    with open('dictionaries.pkl', 'wb') as output:
        pickle.dump(chars_to_indices, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(indices_to_chars, output, pickle.HIGHEST_PROTOCOL)

def load_coded_dictionaries():
    with open('dictionaries.pkl', 'rb') as output:
        chars_to_indices=pickle.load( output)
        indices_to_chars=pickle.load( output)
    return chars_to_indices,indices_to_chars


def encode_text(text,chars_to_indices):   
    """
    Encode a text using a dictionary
    """
    text_coded=np.zeros((len(text)))
    for t, char in enumerate(text):
            text_coded[t]=chars_to_indices[char]
    return text_coded

def decode_text(text_coded,indices_to_chars):
    """
        Encode a text using a dictionary
    """
    text_decoded=""
    for t, index in enumerate(text_coded):
            text_decoded+=indices_to_chars[index]
    return text_decoded
    

if __name__ == "__main__":
    """
    Test the functions to  save and load dictionaries  
    """
    window_size = 100
    step_size = 1
    X, y, chars, chars_to_indices, indices_to_chars,text_clean = LoadText('Beiras.txt', window_size, step_size);
    save_coded_dictionaries(chars_to_indices,indices_to_chars)
    chars_to_indices_new,indices_to_chars_new=load_coded_dictionaries()
    text_org=text_clean[:100]
    print(text_org)
    text_coded=encode_text(text_org,chars_to_indices)
    print(text_coded)
    text_decoded=decode_text(text_coded,indices_to_chars_new)
    print(text_decoded)
    

