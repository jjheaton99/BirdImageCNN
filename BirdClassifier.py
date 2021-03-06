#225 species of birds image classifier
#this is training file, predictions in BirdPredictor.py

#force CPU
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/jj_he/Anaconda3/envs/tensorflow/Library/bin/graphviz'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, concatenate, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from keras.models import Model
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

#--set up the generators--------------------------
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1.0/255)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

#batch size of 25 used because it divides the validation set total
#this allows the entire set to be used for evaluation
train_gen = train_datagen.flow_from_directory(
    directory='train/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=25,
    shuffle=True)
n_train = train_gen.n

valid_gen = valid_datagen.flow_from_directory(
    directory='valid/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=25,
    shuffle=True)
n_valid = valid_gen.n

#moved to prediction file
"""
test_gen = test_datagen.flow_from_directory(
    directory='test/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=25,
    shuffle=False)
n_test = test_gen.n
"""
#--------------------------------------------------

#old VGG style sequential model
"""
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),    
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),   
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(2000, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2000, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(225)
    ])
"""

def convBlock(prev_output, filters=32, kernel_size=(3, 3), num_layers=1, max_pooling=False, pool_size=(3, 3), padding='same'):
    block = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(prev_output)
    block = BatchNormalization()(block)
    if num_layers < 2 and max_pooling:
        block = MaxPooling2D(pool_size=pool_size, padding='same')(block)
    else:
        for i in range(num_layers-1):
            block = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(block)
            block = BatchNormalization()(block)
        if max_pooling:
            block = MaxPooling2D(pool_size=pool_size, padding='same')(block)
    return block

def denseBlock(prev_output, units=10, dropout_rate=0.5, num_layers=1):
    block = Dense(units=units, activation='relu')(prev_output)
    block = BatchNormalization()(block)
    block = Dropout(rate=dropout_rate)(block)
    if num_layers > 1:
        for i in range(num_layers-1):
                block = Dense(units=units, activation='relu')(block)
                block = BatchNormalization()(block)
                block = Dropout(rate=dropout_rate)(block)
    return block

def inceptionBlock(prev_output, filters=32):
    block_1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(prev_output)
    block_1x1 = BatchNormalization()(block_1x1)
    
    block_3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(prev_output)
    block_3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(block_3x3)
    block_3x3 = BatchNormalization()(block_3x3)
    
    block_5x5 = Conv2D(filters, (1, 1), padding='same', activation='relu')(prev_output)
    block_5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(block_5x5)
    block_5x5 = BatchNormalization()(block_5x5)
    
    block_pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_output)
    block_pooling = Conv2D(filters, (1, 1), padding='same', activation='relu')(block_pooling)
    block_pooling = BatchNormalization()(block_pooling)
    
    return concatenate([block_1x1, block_3x3, block_5x5, block_pooling], axis=3)

class InceptionHyperModel(HyperModel):
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        filters1_hp = hp.Int('filters1',
                             min_value=32,
                             max_value=128,
                             step=32,
                             default=32)
        
        filters2_hp = hp.Int('filters2',
                             min_value=32,
                             max_value=128,
                             step=32,
                             default=64)
        
        filters3_hp = hp.Int('filters3',
                             min_value=32,
                             max_value=128,
                             step=32,
                             default=128)
        
        dense_units_hp = hp.Int('dense_units',
                                min_value=256,
                                max_value=2048,
                                step=256,
                                default=512)
        
        dropout_rate_hp = hp.Float('dropout_rate',
                                   min_value=0.0,
                                   max_value=0.9,
                                   step=0.1,
                                   default=0.5)
        
        img_input = keras.Input(shape=self.input_shape)
        
        
        conv_block_1 = convBlock(img_input, filters=filters1_hp, kernel_size=(7, 7), num_layers=1, max_pooling=True)
        conv_block_2 = convBlock(conv_block_1, filters=filters1_hp, kernel_size=(1, 1), num_layers=1, padding='valid')
        conv_block_3 = convBlock(conv_block_2, filters=filters1_hp, num_layers=2, max_pooling=True)
        
        inception_block_1 = inceptionBlock(conv_block_3, filters=filters2_hp)
        inception_block_2 = inceptionBlock(inception_block_1, filters=filters2_hp)
        inception_block_3 = inceptionBlock(inception_block_2, filters=filters2_hp)
        
        average_pool = AveragePooling2D(pool_size=(5, 5))(inception_block_3)
        
        conv_block_4 = convBlock(average_pool, filters=filters3_hp, kernel_size=(3, 3), num_layers=1)
        
        flatten = Flatten()(conv_block_4)
        dense_block1 = denseBlock(flatten, units=dense_units_hp, dropout_rate=dropout_rate_hp, num_layers=2)
        output = Dense(self.num_classes)(dense_block1)
    
        model = Model(inputs=img_input, outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float(
                    'learning_rate',
                    min_value=1e-5,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                    )
                ),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics='accuracy')
        return model
        
def plotLossAccHistory(history):
    fig, axis = plt.subplots(2)
    fig.suptitle('Training losses and accuracies')
    
    axis[0].plot(history.history['loss'], label='loss')
    axis[0].plot(history.history['val_loss'], label='val_loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend(loc='upper right')
    
    axis[1].plot(history.history['accuracy'], label='accuracy')
    axis[1].plot(history.history['val_accuracy'], label='val_accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('accuracy')
    axis[1].legend(loc='upper right')

    plt.show()

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 225

TRAIN_STEP_SIZE = n_train//train_gen.batch_size
VALID_STEP_SIZE = n_valid//valid_gen.batch_size

early_stop = EarlyStopping(monitor='val_loss', patience=20)
model_save = ModelCheckpoint('', save_best_only=True)
learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=10,
                                           verbose=1,
                                           min_lr=1e-5)

hypermodel = InceptionHyperModel(INPUT_SHAPE, NUM_CLASSES)

tuner = Hyperband(hypermodel,
                  objective='val_accuracy',
                  max_epochs=20,
                  directory=os.path.normpath('C:/'),
                  project_name='birds225')

tuner.search(x=train_gen,
             steps_per_epoch=TRAIN_STEP_SIZE,
             epochs=20,
             validation_data=valid_gen,
             validation_steps=VALID_STEP_SIZE)

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters()[0]

print('Optimal hyperparameters:')
print('learning rate: ', best_hps.get('learning_rate'))
print('filters 1: ', best_hps.get('filters1'))
print('filters 2: ', best_hps.get('filters2'))
print('filters 3: ', best_hps.get('filters3'))
print('dense units: ', best_hps.get('dense_units'))
print('dropout rate: ', best_hps.get('dropout_rate'))

model = tuner.hypermodel.build(best_hps)

#model = keras.models.load_model('')

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit(
          x=train_gen,
          steps_per_epoch=TRAIN_STEP_SIZE,
          epochs=1000,
          verbose=2,
          callbacks=[early_stop, model_save, learning_rate_schedule],
          validation_data=valid_gen,
          validation_steps=VALID_STEP_SIZE)
 
model.evaluate(
    valid_gen,
    steps=VALID_STEP_SIZE,
    verbose=2)

plotLossAccHistory(history)