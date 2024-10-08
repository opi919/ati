import numpy as np
from tensorflow.keras.models import Sequential, Model

x = np.arange(-15, 15, 0.03)
y = 5 * x**3 - 8 * x**2 - 7 * x + 1

x = 2 * (x - min(x)) / (max(x) - min(x)) - 1
y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

from sklearn.model_selection import train_test_split

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.14, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

print(len(x_test))

from tensorflow.keras.layers import Dense, Input

inputs = Input(shape=(1,))
x = Dense(32, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(1)(x)
model = Model(inputs, outputs)
model.summary()

import keras
from keras.metrics import R2Score

model.compile(optimizer="adam", loss="mse", metrics=[R2Score(name="accuracy")])
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))


import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Trainning vs Validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()


plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Training vs validaiton error(loss)")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss : {test_loss} and Test accuracy : {test_accuracy}")

y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, label="True values", color="blue")
plt.scatter(x_test, y_pred, label="Predict values", color="red")
plt.title("True vs Predict value")
plt.xlabel("input x")
plt.ylabel("output y")
plt.legend()
plt.show()
