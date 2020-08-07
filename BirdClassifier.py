#225 species of birds image classifier
#this is training file, predictions in BirdPredictor.py

#force CPU
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
#comment out when loading a model
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
#comment out when creating new model
model = tf.keras.models.load_model('',compile=False)

model.summary()

#opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
#good value range found to be 0.005 - 0.0005
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

(loss_ax, val_ax) = plt.subplots(2)
loss_ax.plot(history.history['loss'], label='loss')
loss_ax.plot(history.history['val_loss'], label='val_loss')
loss_ax.xlabel('Epoch')
loss_ax.ylabel('Loss')
loss_ax.legend(loc='upper right')

val_ax.plot(history.history['accuracy'], label='accuracy')
val_ax.plot(history.history['val_accuracy'], label='val_accuracy')
val_ax.xlabel('Epoch')
val_ax.ylabel('accuracy')
val_ax.legend(loc='upper right')

plt.show()