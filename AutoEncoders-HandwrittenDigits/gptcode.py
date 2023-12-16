'''
import numpy as np
import matplotlib.pyplot as plt
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

print(X_train.shape)
#(60000, 784)

# Load and preprocess the MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the images
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Visualize the first 10 training images
plt.figure(figsize=(10, 5))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle('Train Data', fontsize=20)
plt.show()

# Define the autoencoder model
input_dim, encode_dim, hidden_dim, output_dim = 784, 100, 256, 784

input_layer = Input(shape=(input_dim,), name="INPUT")
encoded = Dense(encode_dim, activation='relu', name='BOTTLE_NECK')(input_layer)
decoded = Dense(output_dim, activation='sigmoid', name='OUTPUT')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=5)

# Encode and decode the test data
encoded_data = autoencoder.predict(X_test)

# Visualize the actual, encoded, and decoded data
plt.figure(figsize=(15, 5))

# Actual Data
plt.subplot(1, 3, 1)
plt.imshow(X_test[2].reshape(28, 28), cmap='gray')
plt.title('Actual Data')
plt.axis('off')

# Encoded Data
plt.subplot(1, 3, 2)
plt.imshow(encoded_data[2].reshape(28, 28), cmap='gray')  # Use the same dimensions as for actual data
plt.title('Encoded Data')
plt.axis('off')

# Decoded Data
plt.subplot(1, 3, 3)
plt.imshow(autoencoder.predict(X_test)[2].reshape(28, 28), cmap='gray')  # Decode X_test for visualization
plt.title('Decoded Data')
plt.axis('off')

plt.show()
'''
