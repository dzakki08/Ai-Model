import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from tensorflow import keras


# Matikan optimasi oneDNN agar lebih stabil
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Data contoh (Pendapatan vs Pengeluaran dalam juta rupiah)
X = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]).reshape(-1, 1)  # Pendapatan
Y = np.array([2, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15])  # Pengeluaran

# Normalisasi Data
X_mean, X_std = np.mean(X), np.std(X)
Y_mean, Y_std = np.mean(Y), np.std(Y)

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# --- 1. Regresi Linear ---
lin_reg = LinearRegression()
lin_reg.fit(X_norm, Y_norm)  # Gunakan data yang sudah dinormalisasi
pred_Y_norm = lin_reg.predict(X_norm)

# Denormalisasi hasil prediksi regresi linear
pred_Y = (pred_Y_norm * Y_std) + Y_mean

# Visualisasi Regresi Linear
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, pred_Y, color='red', label='Regresi Linear')
plt.xlabel('Pendapatan (Juta)')
plt.ylabel('Pengeluaran (Juta)')
plt.legend()
plt.title('Regresi Linear')
plt.show()

# --- 2. Two-Layer Neural Network ---
# Model dengan 1 Hidden Layer (10 neuron)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)  # Output Layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
model.fit(X_norm, Y_norm, epochs=100, batch_size=2, verbose=1)

# Prediksi Neural Network
nn_pred_Y_norm = model.predict(X_norm)

# Denormalisasi hasil prediksi dari Neural Network
nn_pred_Y = (nn_pred_Y_norm * Y_std) + Y_mean

# Visualisasi Neural Network
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, nn_pred_Y, color='green', label='Neural Network')
plt.xlabel('Pendapatan (Juta)')
plt.ylabel('Pengeluaran (Juta)')
plt.legend()
plt.title('Prediksi Neural Network')
plt.show()
