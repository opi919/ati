# step - 1 generate trainning data
import numpy as np
from sklearn.model_selection import train_test_split

x = np.arange(-20, 20, 0.1)
y = 5 * x**3 - 8 * x**2 - 7 * x + 1

# normalize data between [-1,1]
x = 2 * (x - min(x)) / (max(x) - min(x)) - 1
y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

print(y.shape)

# step - 3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5 / 95)

# step - 4 reshape
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_val = x_val.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

# step - 5 model architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


input = Input(shape=(1,))
x = Dense(32, activation="relu")(input)
x = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(x)
output = Dense(1)(x)

model = Model(input, output)
model.summary()

# step - 6 compile model
import keras
from keras.metrics import R2Score

model.compile(optimizer="adam", loss="mse", metrics=[R2Score(name="accuracy")])
history = model.fit(
    x_train, y_train, epochs=30, validation_data=(x_val, y_val), batch_size=2
)
print("Trainnig accuracy: ", model.evaluate(x_train, y_train))
print("Validation accuracy: ", model.evaluate(x_val, y_val))
print("Test accuracy: ", model.evaluate(x_test, y_test))
