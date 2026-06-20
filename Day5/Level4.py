print("\n61. Neural networks")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense, Dropout
model = Sequential([Input(shape=(4,)), Dense(8, activation='relu'),Dense(3, activation='softmax')])
model.summary()

print("\n62. Activation functions")
import tensorflow as tf
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
print("ReLU:", tf.nn.relu(x).numpy())
print("Sigmoid:", tf.nn.sigmoid(x).numpy())
print("Tanh:", tf.nn.tanh(x).numpy())

print("\n63. ANN")
model = Sequential([Dense(16, activation='relu', input_shape=(4,)),Dense(8, activation='relu'),Dense(3, activation='softmax')])
model.summary()

print("\n64. Train ANN")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
# Load dataset
iris = load_iris()
X = iris.data
y = to_categorical(iris.target)
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
# Build ANN
model = Sequential([Input(shape=(4,)),Dense(16, activation='relu'),Dense(8, activation='relu'),Dense(3, activation='softmax')])
# Compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Train
model.fit(X_train, y_train, epochs=20)
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

print("\n65. Loss functions")
from tensorflow.keras.losses import MeanSquaredError
y_true = [[1], [2], [3]]
y_pred = [[1.1], [1.8], [3.2]]
mse = MeanSquaredError()
print("Loss:", mse(y_true, y_pred).numpy())

print("\n66. Optimizers")
from tensorflow.keras.optimizers import Adam, SGD
adam = Adam()
sgd = SGD()
print("Adam Optimizer:", adam)
print("SGD Optimizer:", sgd)

print("\n67. Dropout")
model = Sequential([Dense(64, activation='relu', input_shape=(10,)),Dropout(0.5),Dense(1)])
model.summary()


print("\n68. CNN")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input,Conv2D,MaxPooling2D,Flatten,Dense)
model = Sequential([Input(shape=(28,28,1)),Conv2D(32, (3,3), activation='relu'),MaxPooling2D((2,2)),Flatten(),Dense(64, activation='relu'),Dense(10, activation='softmax')])
model.summary()

print("\n70. Transfer learning")
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable = False
print("Transfer Learning Model Loaded")