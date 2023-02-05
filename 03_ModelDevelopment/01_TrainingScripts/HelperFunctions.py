"""
Written by: Rupesh Kumar Dey
Summary: Python Helper function script to develop (train, validate, test) CNN models in stage 1 and stage 2. 
"""

# ================================================================================================================
# IMPORTING REQUIRED LIBRARIES
# ================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy
# import cv2
from PIL import Image
import seaborn as sns
import os
import os.path
from os import path
import sys
import numpy as np
import copy
from copy import deepcopy
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, callbacks, applications
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_hub as hub

# ================================================================================================================
# DATASET CREATION
# ================================================================================================================
def datagen_flow(projectPath, train_path, test_path, batch_size = 32 , image_size = (100,100)):
    """
    Python function to load dataset using tensorflow ImageDataGenerator
    Inputs:
        a) projectPath - project directory
        b) train_path  - folder path where training dataset images are stored
        c) test_path   - folder path where the test / validation dataset images are stored
        d) batch_size  - batch size for training and validation / testing (default 32)
        e) image_size  - input image size (default 100 x 100)

    Outputs:
        a) train_data - batches of training data
        b) valid_data - batches of validation data
        c) test_data - batches of test data
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # Set the seed
    tf.random.set_seed(42)

    # Preprocess the data (get all of the pixel values between 0 & 1, also called scaling / normalization)
    # Generates batches of tensor images
    train_datagen = ImageDataGenerator(rescale = 1./255)    # Set the instance to rescale the image.
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    # Setup paths to dataset directory.
    train_dir = projectPath + train_path
    test_dir = projectPath + test_path

    # Import data from directories and turn it into batches.
    # Also labels the training and validation data
    # Also automatically splits up the data into many mini batches each of size = 32

    # For training data
    train_data = train_datagen.flow_from_directory(directory = train_dir, batch_size = batch_size,
                                                target_size = image_size,       # Reshape to desired image size. Typically image size used is 224 x 224 pizels.
                                                class_mode = "categorical",     # Set to categorical for classification
                                                color_mode = "rgb",             # 3 channels of RGB image
                                                shuffle = "True",               # Shuffle for randomness
                                                seed = 42)                      # Random seed 42

    # For validation / test data
    valid_data = valid_datagen.flow_from_directory(directory = test_dir, batch_size = batch_size,
                                                target_size = image_size,
                                                class_mode = "categorical",
                                                color_mode = "rgb",
                                                shuffle = "True",
                                                seed = 42)
    # For validation / test data
    test_data = test_datagen.flow_from_directory(directory = test_dir, batch_size = 1,
                                                target_size = image_size,
                                                class_mode = "categorical",
                                                color_mode = "rgb",
                                                shuffle = False,
                                                seed = 42)

    # Return batches of training and test / validation data
    return train_data, valid_data, test_data

def kerasImageDataset(projectPath, train_path, test_path, batch_size = 32 , image_size = (100,100)):
    """
    Python function to load dataset using tensorflow image_dataset_from_directory
    Inputs:
        a) projectPath - project directory
        b) train_path  - folder path where training dataset images are stored
        c) test_path   - folder path where the test / validation dataset images are stored
        d) batch_size  - batch size for training and validation / testing (default 32)
        e) image_size  - input image size (default 100 x 100)

    Outputs:
        a) train_data - batches of training data
        b) valid_data - batches of validation data
        c) test_data - batches of test data
    """
    from tensorflow.keras.utils import image_dataset_from_directory
    # Set the seed
    tf.random.set_seed(42)

    # Setup paths to data directory.
    train_dir = projectPath + train_path
    test_dir = projectPath + test_path

    # Import data from directories and turn it into batches.
    # Also labels the training and validation data
    # Also automatically splits up the data into many many batches each of size = 32

    # For training set
    train_data = image_dataset_from_directory(directory = train_dir, batch_size = batch_size,
                                                image_size = image_size,        # Reshape to desired image size. Typically image size used is 224 x 224 pizels.
                                                label_mode = "categorical",     # Categorical for classification problem
                                                color_mode = "rgb",             # Color channels RGB
                                                shuffle = "True",               # Shuffle data
                                                seed = 42)                      # Set seed

    # For test / validation set
    valid_data = image_dataset_from_directory(directory = test_dir, batch_size = batch_size,
                                                image_size = image_size,
                                                label_mode = "categorical",
                                                color_mode = "rgb",
                                                shuffle = "True",
                                                seed = 42)
    # For training / validation set
    test_data = image_dataset_from_directory(directory = test_dir, batch_size = 1,
                                                image_size = image_size,
                                                label_mode = "categorical",
                                                color_mode = "rgb",
                                                shuffle = False,
                                                seed = 42)
    return train_data, valid_data, test_data

# ================================================================================================================
# MODEL ARCHITECTURE DEVELOPMENT: SELF DEVELOPED MODELS
# ================================================================================================================
def createBaseModel(optimizer=None):
    """
    Python function to Create 01_BaseModel1 (Group 1) model architecture
    Inputs:
        a) optimizer - optimizer to be used (None by default)

    Outputs:
        a) base_model - compiled base model
    """
    # import required libraries
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, metrics
    from tensorflow.keras.layers.experimental import preprocessing

    # Build CNN architecture using Keras function API
    inputs = layers.Input(shape=(100, 100, 3), name="input_layer")
    x = layers.Conv2D(8, 3, strides=(1, 1), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x)
    x = layers.Conv2D(16, 3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x)
    x = layers.Conv2D(32, 3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(4, activation="softmax", name="output_layer")(x)
    baseModel = keras.Model(inputs, outputs)

    # Set optimizer (SGD by default if optimizer not defined in input)
    if optimizer == None:
        optimizer = tf.keras.optimizers.SGD()
    else:
        optimizer = optimizer

    # Compile the model
    baseModel.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy", "Recall"])

    # Return compiled model
    return baseModel

def createBaseModel2(input_size=(224, 224, 3), optimizer=None):
    """
    Python function to Create 02_BaseModel2 (Group 2) model architecture
    Inputs:
        a) input_size - define input size (224 x 224 x 3 by default)
        b) optimizer - optimizer to be used (None by default)

    Outputs:
        b) base_model - compiled base model
    """
    # import required libraries
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, metrics
    from tensorflow.keras.layers.experimental import preprocessing

    # Build model architecture using Keras functional API
    inputs = layers.Input(shape=input_size, name="input_layer")
    x = layers.Conv2D(32, 5, strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(4, 4), strides=(2, 2), padding="valid")(x)

    x = layers.Conv2D(16, 5, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x)

    x = layers.Conv2D(8, 3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x)

    x = layers.Conv2D(4, 3, strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1000, activation = "relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(20, activation = "relu")(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(4, activation="softmax", name="output_layer")(x)

    baseModel = keras.Model(inputs, outputs)

    # Set optimizer to SGD by default if not specified by user
    if optimizer == None:
        optimizer = tf.keras.optimizers.SGD()
    else:
        optimizer = optimizer

    # Compile the model
    baseModel.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy", "Recall"])

    # return compiled model
    return baseModel


# ================================================================================================================
# MODEL TRAINING: SELF DEVELOPED MODELS
# ================================================================================================================
def trainModel(baseModel,train_data, valid_data,checkpointPath, epochs = 50, callbacks =  []):
    """
    Python function that trains a created model (basic training with STATIC learning rate regime)
    Inputs:
        a) baseModel - A TF model
        b) train_data - train dataset
        c) valid_data - validation dataset
        d) checkpointPath - folder path to save model checkpoint
        e) epochs - num epochs to train. Default 50
        f) callbacks - External callback functions (if any)

    Outputs:
        a) history_baseModel - model training history object
    """
    # Define steps for traing and validation
    valid_steps = valid_data.n//valid_data.batch_size
    train_steps = train_data.n//train_data.batch_size

    # Create checkpoint callback to save model for later use
    # Saving checkpoint
    checkpoint_pathBaseModel = checkpointPath
    checkpointBaseModel = tf.keras.callbacks.ModelCheckpoint(checkpoint_pathBaseModel,
                                                         save_weights_only=True, # save only the model weights
                                                         monitor="val_accuracy", # save the model weights which score the best validation accuracy
                                                         save_best_only=True, verbose = True) # only keep the best model weights on file (delete the rest)
    callbacks = callbacks.append(checkpointBaseModel)


    # Fit the model
    history_baseModel = baseModel.fit(train_data,
                                  epochs = epochs,
                                  steps_per_epoch = train_steps,
                                  validation_data = valid_data,
                                  validation_steps = valid_steps,
                                  callbacks = [checkpointBaseModel])
    # Return model history object
    return history_baseModel

# Train Model with LrScheduler
def trainModelLrScheduler(baseModel, train_data, valid_data, checkpointPath, epochs=50, epochCutOff = None, reductionRatio = 2 ,callbacks=[]):
    """
    Python function that trains a created model (basic training with DYNAMIC learning rate regime)
    Inputs:
        a) baseModel - A TF model
        b) train_data - train dataset
        c) valid_data - validation dataset
        d) checkpointPath - folder path to save model checkpoint
        e) epochs - num epochs to train. Default 50
        f) epochCutOff - epoch in which the LR starts to reduce (None by default)
        g) reductionRatio - the factor to reduce the LR (2 by default)
        g) callbacks - external callback functions (if any)

    Outputs:
        a) history_baseModel - model training history object
    """
    # Define Training and Validation steps
    valid_steps = valid_data.n // valid_data.batch_size
    train_steps = train_data.n // train_data.batch_size

    # Create checkpoint callback to save model for later use
    # Saving checkpoint
    checkpoint_pathBaseModel = checkpointPath
    checkpointBaseModel = tf.keras.callbacks.ModelCheckpoint(checkpoint_pathBaseModel,
                                                             save_weights_only=True,  # save only the model weights
                                                             monitor="val_accuracy",
                                                             # save the model weights which score the best validation accuracy
                                                             save_best_only=True,
                                                             verbose=True)  # only keep the best model weights on file (delete the rest)

    # Set cutOff point to be half of the total epoch's training if not defined by user (ie total epochs: 100, cutOff: 100 / 2 = 50)
    if epochCutOff == None:
        epochCutOff = epochs / 2
    else:
        pass

    # Defining the learning rate scheduler to control the learning rate of the model
    def scheduler(epoch, lr):    # input of epoch and learning rate (LR)
        if epoch < epochCutOff:  # from epoch 1 to the cutOff epoch, the learning rate is the standard learning rate (SGD - 0.01, Adam - 0.001)
            return lr
        else:                    # After the cutOff epoch, the learning rate is halved with each epoch.
            return lr / reductionRatio

    # Define learning rate callback function
    learningRateCallBack = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    # Fit the model
    history_baseModel = baseModel.fit(train_data,
                                      epochs=epochs,
                                      steps_per_epoch=train_steps,
                                      validation_data=valid_data,
                                      validation_steps=valid_steps,
                                      callbacks=[checkpointBaseModel, learningRateCallBack])

    # Return model training history object
    return history_baseModel

# ================================================================================================================
# MODEL ARCHITECTURE DEVELOPMENT: TRANSFER LEARNING FEATURE EXTRACTION MODEL
# ================================================================================================================
def createHubFeatureExtractionModel(model_url,num_classes = 4, IMAGE_SHAPE = (224,224,3), optimizer = tf.keras.optimizers.SGD()):
    """
    Takes a TensorFlow Hub URL and creates a Keras Sequential model with it. To create a model from a URL from TENSORFLOW HUB.
    Inputs:
        a) model_url - A TensorFlow hub feature extraction URL.
        b) num_classes - Number of output neurons in the output layer, should be equal to number of target classes, default = 4.
        c) IMAGE_SHAPE  - Input image shape into the network (224 x 224 x 3 default)
        d)optimizer - SGD by default (can be specified by user)

    Outputs:
        a) model - transfer learning feature extraction model
    """
    # Download the pretrained model and save it as a Keras Layer.
    feature_extractor_layer = hub.KerasLayer(model_url,                   # Put this entire into a Keras Sequential mode layer
                                           trainable = False,             # Freeze the already learned patterns from ImageNet dataset. Ie. the already trained hidden layers of the pretrained model will not be trained.
                                           name = "feature_extraction_layer",
                                           input_shape = IMAGE_SHAPE)   # IMAGE_SHAPE is defined as (224,224) by defaul.

    # Add on additional dense layers to the feature extraction model
    model = tf.keras.Sequential([
      feature_extractor_layer,                # The transfer learning model
      layers.BatchNormalization(),            # Top up the additional dense layers
      layers.Dense(1000, activation="relu"),
      layers.Dropout(0.2),
      layers.BatchNormalization(),
      layers.Dense(20, activation="relu"),
      layers.BatchNormalization()(x),
      layers.Dense(num_classes,activation = "softmax",name = "output_layer")
    ])

    # Compile model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy","Recall"])

    # Return compiled model
    return model

def createApplicationsFeatureExtractionModel(model, trainable = False ,num_classes = 4, IMAGE_SHAPE = (224,224,3), optimizer = tf.keras.optimizers.SGD()):
    """
    Takes a transfer learning model from TENSORFLOW APPLICATIONS API and creates a Keras Sequential feature extraction model with it.
    Inputs:
        a) model - A TensorFlow model created using TENSORFLOW APPLICATIONS API.
        b) trainable - Set whether to lock / unlock base model layers for training (False by default)
        c) num_classes - Number of output neurons in the output layer, should be equal to number of target classes, default = 4.
        d) IMAGE_SHAPE  - Input image shape into the network (224 x 224 x 3 default)
        e)optimizer - SGD by default (can be specified by user)

    Outputs:
        a) finalModel - transfer learning feature extraction model
    """
    # Create base model which is the transfer learning model
    baseModel = model
    if trainable == False:
        baseModel.trainable = False
    else:
        pass

    # Create an input layer
    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")

    # Build the model (Adding on additional dense layers)
    x = baseModel(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1000, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Create the final output layer
    outputs = layers.Dense(4, activation="softmax", name="output_layer")(x)

    finalModel = keras.Model(inputs, outputs)

    if optimizer == None:
        optimizer = tf.keras.optimizers.SGD()
    else:
        optimizer = optimizer

    # Compile the model
    finalModel.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy", "Recall"])
    # Return compiled model
    return finalModel

def createApplicationsFineTuningModel(model, trainable = True, numLayersUnfreeze = 10 ,num_classes = 4, IMAGE_SHAPE = (224,224,3), optimizer = tf.keras.optimizers.SGD()):
    """
    Takes a transfer learning model from TENSORFLOW APPLICATIONS API and creates a Keras Sequential fine tuning model with it.
    Inputs:
        a) model - A TensorFlow model created using TENSORFLOW APPLICATIONS API.
        b) trainable - Set whether to lock / unlock base model layers for training (True by default)
        c) numLayersUnfreeze - Set the number of layers to unlock in the base model for training (10 by default)
        d) num_classes - Number of output neurons in the output layer, should be equal to number of target classes, default = 4.
        e) IMAGE_SHAPE  - Input image shape into the network (224 x 224 x 3 default)
        f)optimizer - SGD by default (can be specified by user)

    Outputs:
        a) finalModel - transfer learning fine tuning model
    """
    # Create base model which is the transfer learning model. Unlock final n layers for training
    baseModel = model
    if trainable == False:
        baseModel.trainable = False
    else:
        baseModel.trainable = True
        for layer in baseModel.layers[:-numLayersUnfreeze]:
          layer.trainable = False

    # Create an input layer
    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")

    # Build the model
    x = baseModel(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1000, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Create the final output layer
    outputs = layers.Dense(4, activation="softmax", name="output_layer")(x)

    finalModel = keras.Model(inputs, outputs)

    # Set optimizer
    if optimizer == None:
        optimizer = tf.keras.optimizers.SGD()
    else:
        optimizer = optimizer

    # Compile the model
    finalModel.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy", "Recall"])
    # Return compiled model
    return finalModel

# ================================================================================================================
# MODEL ARCHITECTURE CHECKPOINT LOADING
# ================================================================================================================
def loadModel(baseModel, checkpointPath):
    """
    Function to load saved model from checkpoint folder
    Inputs:
      a) baseModel      - Base model that is similar structure to the saved model
      b) checkpointPath - path to load model weights from

    Outputs:
      a) baseModel with loaded weights
    """
    # Load best model from checkpoint folder
    checkpoint_pathBaseModel = checkpointPath
    baseModel.load_weights(checkpoint_pathBaseModel)
    return baseModel

# ================================================================================================================
# MODEL TESTING
# ================================================================================================================
def testResults(baseModel, test_data, datatype, data_class=False, location = None):
    """
    Function to Test Model's performance on a test dataset
    Function that returns the prediction confusion matrix and classification report in dataframe for a given model and test set
    Inputs:
        a) baseModel - base trained model
        b) test_data - testing data used to test the model
        c) datatype - (string input) options of [noIPCV, CLAHE, MULTISCALE, RAYLEIGH]
        d) data_class - default is False but user should input in array the list of classes ie. [0_normal_(IPCV METHOD),1_.....]

    Outputs:
        a) df - classification report dataframe
        b) confusionMatrix - confusion matrix
        c) y_true - Actual class
        d) predictions - Predicted class
    """
    # Import required library
    from sklearn.metrics import confusion_matrix, classification_report

    # Define step size
    STEP_SIZE_TEST = test_data.n // test_data.batch_size
    test_data.reset()

    # Run prediction
    pred = baseModel.predict(test_data, steps=STEP_SIZE_TEST, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    # Fixing prediction labels accordingly
    labels = (test_data.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[v] for v in predicted_class_indices]

    # Default label if label not specified by user
    if data_class == False:
        data_class = ["0_normal", "1_ulcerative_colitis", "2_polyps", "3_esophagitis"]
    else:
        data_class = data_class

    # Aligning class labels between predictions and actual resutls
    # Extract test results
    test_results = test_data.classes
    test_results
    class_labels = []
    for label in test_results:
        class_labels.append(label)

    # Ammend class labels for y_true
    y_true = deepcopy(class_labels)
    for i in range(len(class_labels)):
        if int(class_labels[i]) == 0:
            y_true[i] = data_class[0]
        elif int(class_labels[i]) == 1:
            y_true[i] = data_class[1]
        elif int(class_labels[i]) == 2:
            y_true[i] = data_class[2]
        elif int(class_labels[i]) == 3:
            y_true[i] = data_class[3]

    # Create confusion matrix
    confusionMatrix = confusion_matrix(y_true, predictions,
                                       labels=data_class)
    # Plots confusion matrix
    print("Confusion Matrix:")
    plt.figure()
    sns.heatmap(confusionMatrix,
                xticklabels=["predicted_0_normal",
                             "predicted_1_ulcerative_colitis",
                             "predicted_2_polyps",
                             "predicted_3_esophagitis"]
                , yticklabels=["actual_0_normal",
                               "actual_1_ulcerative_colitis",
                               "actual_2_polyps",
                               "actual_3_esophagitis"]
                , annot=True, fmt='.2f')

    plt.title("Confusion Matrix " + datatype)

    # Save confusion matrix at specified location to save results
    if location != None:
        checkPath, fileExtension = os.path.split(location)[0],os.path.split(location)[1]
        if os.path.isdir(checkPath) == False:
            os.makedirs(checkPath)
        else:
            pass
        plt.savefig(location,bbox_inches='tight')
    else:
        pass

    # Return dataframe of classification report
    df = pd.DataFrame(classification_report(y_true, predictions, output_dict=True))
    return df, confusionMatrix, y_true, predictions

# ================================================================================================================
# SAVING CLASSIFICATION REPORT
# ================================================================================================================
def saveResults_csv(df,location = None):
    """
    Function to save df into csv in a nested folder
    Inputs:
        a) df - Classification report DataFrame
        b) location - location to save the dataframe in csv format
    """
    if location != None:
        checkPath, fileExtension = os.path.split(location)[0],os.path.split(location)[1]
        if os.path.isdir(checkPath) == False:
            os.makedirs(checkPath)
        else:
            pass
        df.to_csv(location)
    else:
        pass

# ================================================================================================================
# PLOT MODEL TRAINING HISTORY
# ================================================================================================================
# Function to plot training history
def plot_history(history, testTitle, location = None):
    """
    Function to plot and save model training history
    Inputs:
        a) history - training history object
        b) testTitle - Title for plotting in figures
        c) location - location to save history graph plot
    """
    import matplotlib.pyplot as plt

    # Plotting training history
    plt.figure(figsize = (20,10))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],label = 'training_accuracy')
    plt.plot(history.history['val_accuracy'],label = 'testing_accuracy')
    plt.title("Model Accuracy over Epochs " + testTitle)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['recall'],label = 'training_recall')
    plt.plot(history.history['val_recall'],label = 'testing_recall')
    plt.title("Model Recall over Epochs " + testTitle)
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()

    # Saving history plot
    if location != None:
      checkPath, fileExtension = os.path.split(location)[0], os.path.split(location)[1]
      if os.path.isdir(checkPath) == False:
          os.makedirs(checkPath)
      else:
          pass
      plt.savefig(location,bbox_inches='tight')
    else:
      pass

# ================================================================================================================
# SAVE AND LOAD MODEL TRAINING HISTORY
# ================================================================================================================
def save_history1(history, location):
    """
    Function to save model training history (as pickle object)
    Inputs:
        a) history - training history object
        b) location - location to save training history
    """
    # Save model traning history at specified location. If not save at default save location
    if location != None:
        checkPath, fileExtension = os.path.split(location)[0], os.path.split(location)[1]
        if os.path.isdir(checkPath) == False:
            os.makedirs(checkPath)
        else:
            pass
    np.save(location, history)

def load_history(location):
    """
    Function to load model training history
    Inputs:
        a) location - location to load the model training history from

    Outputs:
        b) history - loaded model training history
    """
    # Load model History
    history = np.load(location, allow_pickle='TRUE').item()
    # Return loaded history
    return history

# ================================================================================================================
# SUMMARIZING CLASSIFICATION REPORTS
# ================================================================================================================
def extractClassificationReport(reportsPath):
    """
    Python function file to extract test and validation data from results folders with classification reports and summarize the results in a single CSV file.
    Inputs:
        a) reportsPath - Location where classification reports are stored


    Outputs:
        a) dfResults - summarized results
    """
    import os
    import pandas as pd

    # Create template dataframe
    dfResults = pd.DataFrame(columns=['modelName', 'modelGroup', 'inputSize', 'optimizer', 'lrType',
                                      'transferLearning', 'featureExtraction', 'fineTuning', 'augmentation',
                                      'valAccuracy', 'valPrecision', 'valF1Score', 'testAccuracy', 'testPrecision',
                                      'testF1Score',
                                      'normal_recall_val', 'ulcerative_colitis_recall_val', 'polyps_recall_val',
                                      'esophagitis_recall_val',
                                      'normal_recall_test', 'ulcerative_colitis_recall_test', 'polyps_recall_test',
                                      'esophagitis_recall_test', 'ValTestAverage'])

    # Specify location of test and validation files
    fileListVal = os.listdir(reportsPath + "val/")
    fileListTest = os.listdir(reportsPath + "test/")

    # Ensure that the same number of files for validation and test are saved
    assert len(fileListVal) == len(
        fileListTest), "Please check file directory and ensure both test and val folders have a complete set of data"

    # Process each file
    for i in range(len(fileListVal)):
        # Filename
        filename, file_extension = os.path.splitext(fileListVal[i])

        # Test and validation file to read data from to summarize
        dfVal = pd.read_csv(reportsPath + "val/" + fileListVal[i])
        dfTest = pd.read_csv(reportsPath + "test/" + filename + "_TEST.csv")

        # Print path to check
        print(reportsPath + "val/" + fileListVal[i])
        print(reportsPath + "test/" + filename + "_TEST.csv")
        print("\n")

        # Extract data
        modelName = os.path.splitext(fileListVal[i])[0]
        modelID = dfVal["modelTag"][0]

        # Input size
        if "basemodel1" in modelName.lower():
            inputSize = 100
        else:
            inputSize = 224

        # optimizer
        if "adam" in modelName.lower():
            optimizer = "ADAM"
        else:
            optimizer = "SGD"

        # Learning rate regime
        if "lrscheduler" in modelName.lower():
            lrType = "DYNAMIC"
        else:
            lrType = "STATIC"

        # Transfer learning implemented or not
        if "transferlearning" in modelName.lower():
            transferLearning = True
        else:
            transferLearning = False

        # Is transfer learning type a feature extraction type
        if "transferlearningfeatureextraction" in modelName.lower() and transferLearning == True:
            featureExtraction = True
        else:
            featureExtraction = False

        # Is transfer learning type a fine tuning type
        if "transferlearningfinetuning" in modelName.lower() and transferLearning == True:
            fineTuning = True
        else:
            fineTuning = False

        # Is the model trained on augmented data
        if "11_" in modelName.lower():
            augmentation = True
        else:
            augmentation = False

        # Extract results.
        valAccuracy = dfVal["macro avg"][1]
        valPrecision = dfVal["macro avg"][0]
        valF1Score = dfVal["macro avg"][2]

        testAccuracy = dfTest["macro avg"][1]
        testPrecision = dfTest["macro avg"][0]
        testF1Score = dfTest["macro avg"][2]

        normal_recall_val = dfVal[dfVal.columns[1]][1]
        ulcerative_colitis_recall_val = dfVal[dfVal.columns[2]][1]
        polyps_recall_val = dfVal[dfVal.columns[3]][1]
        esophagitis_recall_val = dfVal[dfVal.columns[4]][1]

        normal_recall_test = dfTest[dfTest.columns[1]][1]
        ulcerative_colitis_recall_test = dfTest[dfTest.columns[2]][1]
        polyps_recall_test = dfTest[dfTest.columns[3]][1]
        esophagitis_recall_test = dfTest[dfTest.columns[4]][1]

        # Append data to dataframe
        dfResults = dfResults.append({
            'modelName': modelName,
            'modelGroup': modelID,
            'inputSize': inputSize,
            'optimizer': optimizer,
            'lrType': lrType,
            'transferLearning': transferLearning,
            'featureExtraction': featureExtraction,
            'fineTuning': fineTuning,
            'augmentation': augmentation,
            'valAccuracy': valAccuracy,
            'valPrecision': valPrecision,
            'valF1Score': valF1Score,
            'testAccuracy': testAccuracy,
            'testPrecision': testPrecision,
            'testF1Score': testF1Score,
            'normal_recall_val': normal_recall_val,
            'ulcerative_colitis_recall_val': ulcerative_colitis_recall_val,
            'polyps_recall_val': polyps_recall_val,
            'esophagitis_recall_val': esophagitis_recall_val,
            'normal_recall_test': normal_recall_test,
            'ulcerative_colitis_recall_test': ulcerative_colitis_recall_test,
            'polyps_recall_test': polyps_recall_test,
            'esophagitis_recall_test': esophagitis_recall_test,
            'ValTestAverage': 0.5 * (valAccuracy + testAccuracy)},
            ignore_index=True)

    return dfResults



# =================================================================================================