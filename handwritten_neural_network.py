import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
# NOTE: Load dataset of 70000 examples of handwritten digits.
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# NOTE: Normalize data to make it easier to process.
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# NOTE: Building Neural Network.
model = tf.keras.models.Sequential()

# NOTE: ADD Layers.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Input Layer.
# NOTE: ADD 'hidden' layers.
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# NOTE: ADD 'output' layer.
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# NOTE: Compiling the Model.
# - `compile` method is called on the `model` object for compiling it before
# the training;
# - `optimizer='adam'` the optimizer specifies which algorithm use to adjust
# the weighs of the model;
# - `loss='sparse_categorical_crossentropy` the loss function specifies how
# compute the error between the predictions and desired output values.
# - `metrics=['accuracy']` specify the model evaluation measures that you want
# to compute during training and evaluation.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# NOTE: Training & Testing the Model.
# epochs = How many times the model is going to see the same data
# over and over again.
model.fit(X_train, y_train, epochs=3)
loss, accuracy = model.evaluate(X_test, y_test)

print(loss)
print(accuracy)



model.save('digits.model')

"""

# 1- Loading a pre-trained model
model = tf.keras.models.load_model('digits.model')
# 2.1- `cv2.imread()` load the image from the specific file and returns a
# 'numpy matrix' 3x3;
# 2.2- `[:, :, 0]`, we pick all the values of the rows, all the values of the
# columns and '0' because we want to take only the rows and columns of the blue
# channel;
img = cv2.imread('digit.png')[:, :, 0]
# `invert` method takes the img via two-dimensional array [img], because this
# method works only with array, then it inverts the colour of the pixel;
img = np.invert(np.array([img]))
prediction = model.predict(img)
print("Prediction: {}".format(np.argmax(prediction)))
plt.imshow(img[0])
plt.show()
