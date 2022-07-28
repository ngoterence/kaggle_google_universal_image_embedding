import tensorflow as tf
from scipy.ndimage import zoom
import numpy as np
#import tensorflow_addons as tfa
from data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

BASE_PATH = "data/base_pictures/"
AUGMENTED_PATH = "data/augmented_data/"
HEIGHT, WIDTH = 224, 224

def resize_img(img, new_height=224, new_width=224):
    h,w = img.shape[:2]
    img_resized = zoom(img, (new_height/h, new_width/w, 1))
    return img_resized


def model_simple():
    image = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
    output = tf.cast(image, tf.float32)
    output = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [224, 224]), name='resize')(output)

    #output = tf.keras.applications.inception_v3.preprocess_input(output)
    #output = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=[None, None, 3],
                                                            #include_top=False, pooling='avg')(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(64, name='embedding')(output)
    #output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='embedding_norm')(output)
    model = tf.keras.Model(inputs=[image], outputs=[output])

    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    #tf.saved_model.save(model, 'inceptionv3/')
    return model


if __name__=="__main__":
    # load datasets
    # ...

    model = model_simple()

    img = np.zeros((1, 224, 224, 3))
    pred = model(img)
    print(pred.shape)


