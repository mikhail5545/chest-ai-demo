import tensorflow as tf
import pandas as pd


def neural_model(train_x: pd.DataFrame) -> tf.keras.models.Sequential:
    try:
        with tf.device('/device:GPU:0'):
            core_model = tf.keras.models.Sequential()
            core_model.add(tf.keras.applications.mobilenet.MobileNet(input_shape=train_x.shape[1:], include_top=False, weights=None))
            core_model.add(tf.keras.layers.GlobalAveragePooling2D())
            core_model.add(tf.keras.layers.Dropout(0.5))
            core_model.add(tf.keras.layers.Dense(512))
            core_model.add(tf.keras.layers.Dropout(0.5))
            core_model.add(tf.keras.layers.Dense(15, activation='tanh'))
            core_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'mse'])
    except RuntimeError as e:
        print(e)
    
    print(core_model.summary())


    return core_model
