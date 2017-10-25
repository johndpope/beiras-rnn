import numpy as np
import re
import pickle

def window_transform_text(text,window_size,step_size):
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

# transform character-based input/output into equivalent numerical versions
def encode_io_pairs(text,window_size,step_size,chars_to_indices):
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

def LoadText(sz_file,window_size,step_size):
    text_org = open(sz_file, encoding="utf-8").read().lower()
    text_without_source="";
    regexp=re.compile(r'http')
    for line in text_org.splitlines():
        if not regexp.search(line):
            text_without_source= text_without_source + line
    text_clean = re.sub('[ºªàâäçèêïìôöü&%@•…«»”“*/!"(),.:;_¿¡¿‘’´\[\]\']',' ',text_without_source)
    text_clean = text_clean.replace("  "," ")
    chars=sorted(list(set(text_clean )))
    # this dictionary is a function mapping each unique character to a unique integer
    chars_to_indices = dict((c, i) for i, c in enumerate(chars))  # map each unique character to unique integer

    # this dictionary is a function mapping each unique integer back to a unique character
    indices_to_chars = dict((i, c) for i, c in enumerate(chars))  # map each unique integer back to unique character
    X,y = encode_io_pairs(text_clean,window_size,step_size,chars_to_indices)
    return X,y,chars,chars_to_indices,indices_to_chars


def predict_next_chars(model,input_chars,num_to_predict,chars_to_indices,indices_to_chars):     
    # create output
    predicted_chars = ''
    for i in range(num_to_predict):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, window_size, len(chars)))
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


def print_predicctions(model,weights_file,num_to_predict,chars_to_indices,indices_to_chars):
    start_inds = [100,1000,5000]

    # load in weights
    model.load_weights(weights_file)
    for s in start_inds:
        start_index = s
        input_chars = text_clean[start_index: start_index + window_size]

        # use the prediction function
        predict_input = predict_next_chars(model,input_chars,num_to_predict = 100)

        # print out input characters
        print('------------------')
        input_line = 'input chars = ' + '\n' +  input_chars + '"' + '\n'
        print(input_line)

        # print out predicted characters
        line = 'predicted chars = ' + '\n' +  predict_input + '"' + '\n'
        print(line)  

def save_coded_dictionaries(chars_to_indices,indices_to_chars):
    with open('dictionaries.pkl', 'wb') as output:
        pickle.dump(chars_to_indices, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(indices_to_chars, output, pickle.HIGHEST_PROTOCOL)

def load_coded_dictionaries(chars_to_indices,indices_to_chars):
    with open('dictionaries.pkl', 'wb') as output:
        pickle.loca(chars_to_indices, output, pickle.HIGHEST_PROTOCOL)
        pickle.load(indices_to_chars, output, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    window_size = 100
    step_size = 1
    X, y, chars, chars_to_indices, indices_to_chars = LoadText('Beiras.txt', window_size, step_size);




