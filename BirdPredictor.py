#Prediction of a single image using bird classifier

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2

train_datagen = ImageDataGenerator(
    vertical_flip=True,
    rescale=1.0/255)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    directory='train/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=25,
    shuffle=True)
TRAIN_STEP_SIZE = train_gen.n//train_gen.batch_size

test_gen = test_datagen.flow_from_directory(
    directory='test/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=25,
    shuffle=False)
TEST_STEP_SIZE = test_gen.n//test_gen.batch_size

model = tf.keras.models.load_model('',compile=True)

model.evaluate(
    test_gen,
    steps=TEST_STEP_SIZE,
    verbose=2)

#--Test set predictions--------------------------
test_gen.reset()
predictions = model.predict(test_gen, 
                            steps=TEST_STEP_SIZE,
                            verbose=1)

pred_classes = np.argmax(predictions, axis=1)

labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in pred_classes]

filenames = test_gen.filenames
results = pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
#------------------------------------------------

#outputs prediction for image file and associated confidence
def predictImage(filename, model=model, labels=labels):
    image = cv2.imread(filename)/255.0
    #expand to 4D tensor so it fits the batch shape
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image, steps=1, verbose=1)
    pred_tensor = tf.constant(prediction)
    probs = tf.keras.activations.softmax(pred_tensor).numpy()
    pred_class = np.argmax(probs, axis=1)
    return ([labels[k] for k in pred_class], probs[0][pred_class[0]])
    
print(predictImage('testbird.jpg'))