
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataAugmentationFolder
from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import WOAbnormalImages
from DDSM_2_Folders import WOAbnormalMassImages

from DDSM_2_Folders import NOAbnormalImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import CLAHEAbnormalImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

from DDSM_2_Folders import DataAugmentationFolder

from DDSM_2_Folders import NONormalImages

from DDSM_7_1_CNN_Architectures import MobileNetV3Large_Pretrained

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_4_DDSM_Functions import BiclassPrinting
from DDSM_6_Data_Augmentation import DataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
"""
Calcification = 3
Mass = 2

NCalcification = 'Calcification'
NMass = 'Mass'

IN = 0 
IT = 1

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_Calcification, Labels_Calcification = DataAugmentation(WOAbnormalImages, DataAugmentationFolder, NCalcification, Calcification, IN)

Images_Mass, Labels_Mass = DataAugmentation(WOAbnormalMassImages, DataAugmentationFolder, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NOImages_Calcification, NOLabels_Calcification = DataAugmentation(NOAbnormalImages, NCalcification, Calcification, IN)

NOImages_Mass, NOLabels_Mass = DataAugmentation(NOAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CLAHEImages_Calcification, CLAHELabels_Calcification = DataAugmentation(CLAHEAbnormalImages, NCalcification, Calcification, IN)

CLAHEImages_Mass, CLAHELabels_Mass = DataAugmentation(CLAHEAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BiclassPrinting(Images_Calcification, Images_Mass, RAWTechnique)
BiclassPrinting(NOImages_Calcification, NOImages_Mass, NOTechnique)
BiclassPrinting(CLAHEImages_Calcification, CLAHEImages_Mass, CLAHETechnique)

"""


import os
  
train_dir = r'D:\DDSM\NoNormalSplitData\train'
valid_dir = r'D:\DDSM\NoNormalSplitData\val'
test_dir = r'D:\DDSM\NoNormalSplitData\test'

#train_data_dir = DataAugmentationFolder + '/Mammography_data/train'
#validation_data_dir = DataAugmentationFolder + '/Mammography_data/validation'
#test_data_dir = DataAugmentationFolder + '/Mammography_data/test'

train_datagen = ImageDataGenerator(
    zoom_range = 0.2,
    rotation_range = 70,
    horizontal_flip = True,
    vertical_flip = True)

validation_datagen = ImageDataGenerator(
    zoom_range = 0.2,
    rotation_range = 70,
    horizontal_flip = True,
    vertical_flip = True)

test_datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (244, 244),
    batch_size = 16,
    class_mode = 'categorical')

valid_generator = train_datagen.flow_from_directory(
    valid_dir,
    target_size = (244, 244),
    batch_size = 16,
    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (244, 244),
    batch_size = 16,
    class_mode = 'categorical')

def build_model():
    base_model = densenet.DenseNet121(input_shape=(224, 224, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')
    for layer in base_model.layers:
      layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

model_history = model.fit_generator(
    train_generator,
    epochs=20,
    validation_data = valid_generator,
    validation_steps = 307 // 16,
    callbacks=callbacks_list)