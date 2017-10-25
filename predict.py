
import sys
from beiras_aux import load_coded_dictionaries,predict_next_chars,clean_text
from keras.layers import Dense, Activation,GRU
from keras.models import Sequential

window_size = 100


def create_gru_model(chars):
    num_chars = len(chars)
    model= Sequential()
    # 1 Layer .- LSTM layer 1 should be an LSTM module with 200 hidden units
    model.add(GRU(200,input_shape = (window_size,num_chars),return_sequences=True))
    # 2 Layer .-  Dense, with number chars unit and softmax activation
    model.add(GRU(200))
    model.add(Dense(num_chars,activation='softmax'))
    return model


    

def predict(sentence):
    chars_to_indices,indices_to_chars=load_coded_dictionaries()
    model=create_gru_model(chars_to_indices);
    model.load_weights('model_weights/best_beiras_gru_textdata_weights.hdf5')
    return predict_next_chars(model,sentence,window_size,chars_to_indices,indices_to_chars)



if __name__ == "__main__":
    input_sentence=' '.join(sys.argv[1:])
    input_sentence=clean_text(input_sentence.lower())
    if (len(input_sentence)<window_size):
        print("Sentence must have ",window_size,len(input_sentence));
        sys.exit(0)
    input_sentence=input_sentence[:window_size]
    
    predicted=predict(input_sentence)
    print(input_sentence,"...",predicted);
      