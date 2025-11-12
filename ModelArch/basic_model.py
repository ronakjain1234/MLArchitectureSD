import tensorflow as tf
from tensorflow.keras import layers, models

# ========================================================== #
# Some potential hyperparameters to search over. We will expand this as
# we get more data.
# ========================================================== #

search_space = {
    "num_conv_layers": [1, 2],
    "filters": [4, 8, 16],
    "kernel_size": [3, 5],
    "depthwise": [False, True],# better for contrained hardware
    "dense_units": [4, 8, 16],
    "activation": ["relu"],
    "pooling": ["max", "avg"],
}


# ========================================================== #
# Basic Convolutional Model. We will update this model later when 
# we get data. We plan on allocating 50KB of RAM for the model, so 
# we have a lot of room to grow. 
# ========================================================== #

def make_baseline_model(window_len=100, n_Axes=3, n_classes=10):
    """
    Total Params: 970 (3.79 KB)
    Trainable Params: 970 (3.79 KB)
    Non-Trainable Params: 0 (0.00 KB)
    """
    inputs = layers.Input(shape=(window_len, n_Axes))

    # Encoder (Conv1d layers)
    x = layers.Conv1D(filters = 8, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters = 16, kernel_size=3, activation='relu', padding='same')(x)
    latent = layers.GlobalAveragePooling1D()(x) # latent vector 

    # Classifier Branch
    classifier = layers.Dense(16, activation='relu')(latent)
    classifier_output = layers.Dense(n_classes, activation='softmax')(classifier)


    # Decoder 
    x_dec = layers.Reshape((1,16))(latent)
    x_dec = layers.UpSampling1D(size=window_len // 1)(x_dec)  
    x_dec = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x_dec)
    recon_output = layers.Conv1D(filters=n_Axes, kernel_size=5, activation='linear', padding='same', name='recon_output')(x_dec)

    # Final Model
    model = models.Model(inputs=inputs, outputs=[classifier_output, recon_output])

    return model




    



model = make_baseline_model(window_len=100, n_Axes=3, n_classes=10)
model.summary()

model.compile(
    optimizer='adam',                 
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']              
)

history = model.fit(
    X_train,         
    y_train,          
    epochs=20,        
    batch_size=32,    
    validation_data=(X_val, y_val)  
)

