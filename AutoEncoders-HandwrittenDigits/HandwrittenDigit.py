import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

(X_train,_),(X_test, _) = mnist.load_data()

print(X_train.shape)
#(60000, 28, 28)

X_train = X_train/255.0
X_test = X_test/255.0

X_train = X_train.reshape(len(X_train),28*28)
X_test = X_test.reshape(len(X_test),28*28)
'''X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)'''


print(X_train.shape)
#(60000, 784)


def show_visual(data, title, n=10, height=28, width=28):
  plt.figure(figsize=(10,5))
  for i in range(n):
    ax = plt.subplot(1,n,i+1)
    plt.imshow(data[i].reshape(height,width))
    plt.gray() 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.suptitle(title, fontsize=20)
  plt.show()

show_visual(X_train,title='Train Data')
show_visual(X_test,title='Test Data')

input_dim, output_dim = 784, 784
encode_dim = 100
hidden_dim = 256

# Encoder
input_layer = Input(shape=input_dim, name="INPUT")
hidden_layer_1 = Dense(hidden_dim, activation='relu', name='HIDDEN_1')(input_layer)

# Bottle Neck
bottle_neck = Dense(encode_dim, activation='relu', name='BOTTLE_NECK')(hidden_layer_1)

# Decoder
hidden_layer_2 = Dense(hidden_dim, activation='relu', name='HIDDEN_2')(bottle_neck)
output_layer = Dense(output_dim, activation='sigmoid', name='OUTPUT')(hidden_layer_2)



model = Model(input_layer, output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train,X_train, epochs=5)

decoded_data = model.predict(X_test)

get_encoded_data = Model(inputs=model.input,
                         outputs = model.get_layer('BOTTLE_NECK').output)


encoded_data = get_encoded_data.predict(X_test)

show_visual(X_test, title="Actual Data")
show_visual(encoded_data, title="Encoded Data", height=10, width=10)
show_visual(decoded_data, title="Decoded Data")
