import tensorflow as tf


def build_fpn_inception_model():
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.Conv2D(
        64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs, x)
