import tensorflow as tf
from tensorflow.keras import layers, models

def make_baseline_model(window_len=8, n_Axes=3, n_classes=10):
    inputs = layers.Input(shape=(window_len, n_Axes))
    x = layers.Conv1D(filters = 8, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters = 16, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=out)
    return model