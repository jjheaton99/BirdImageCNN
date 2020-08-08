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

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import Model

#--set up the generators--------------------------
train_datagen = ImageDataGenerator(
    vertical_flip=True,
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

def convBlock(prev_output, filters=32, kernel_size=(3, 3), num_layers=1, max_pooling=False, pool_size=(2, 2)):
    block = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(prev_output)
    block = BatchNormalization()(block)
    if num_layers < 2 and max_pooling:
        block = MaxPooling2D(pool_size=pool_size, padding='same')(block)
    else:
        for i in range(num_layers-1):
            block = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(block)
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

img_input = keras.Input(shape=(224, 224, 3))

conv_block_1 = convBlock(img_input, filters=32, num_layers=2, max_pooling=True)

inception_block_1 = inceptionBlock(conv_block_1, filters=32)
inception_block_2 = inceptionBlock(inception_block_1, filters=64)
inception_block_3 = inceptionBlock(inception_block_2, filters=128)

conv_block_2 = convBlock(inception_block_3, filters=128, num_layers=2, max_pooling=True)
conv_block_3 = convBlock(conv_block_2, filters=256, num_layers=2)

global_average = GlobalAveragePooling2D()(conv_block_3)
dense_block1 = denseBlock(global_average, units=2000, dropout_rate=0.5, num_layers=2)
output = Dense(225)(dense_block1)

model = Model(inputs=img_input, outputs=output)

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
#model = tf.keras.models.load_model('',compile=False)

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
"""
#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#good learning rate range found to be 0.005 - 0.0005
opt = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics='accuracy')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model_save = ModelCheckpoint('', save_best_only=True)

TRAIN_STEP_SIZE = n_train//train_gen.batch_size
VALID_STEP_SIZE = n_valid//valid_gen.batch_size

history = model.fit(
          x=train_gen,
          steps_per_epoch=TRAIN_STEP_SIZE,
          epochs=1000,
          callbacks=[early_stop, model_save],
          validation_data=valid_gen,
          validation_steps=VALID_STEP_SIZE)
 
model.evaluate(
    valid_gen,
    steps=VALID_STEP_SIZE,
    verbose=2)

fig, axis = plt.subplots(2)
fig.suptitle('Training losses and accuracies')

axis[0].plot(history.history['loss'], label='loss')
axis[0].plot(history.history['val_loss'], label='val_loss')
axis[0].xlabel('Epoch')
axis[0].ylabel('Loss')
axis[0].legend(loc='upper right')

axis[1].plot(history.history['accuracy'], label='accuracy')
axis[1].plot(history.history['val_accuracy'], label='val_accuracy')
axis[1].xlabel('Epoch')
axis[1].ylabel('accuracy')
axis[1].legend(loc='upper right')

plt.show()
"""