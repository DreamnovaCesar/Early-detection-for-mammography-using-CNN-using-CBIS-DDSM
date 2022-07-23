
import numpy as np

from CBIS_DDSM_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Biclass_ML(Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    #Iter_Mass = 20 
    #Iter_tumor = 40 

    #Iter_mass = 5 
    #Iter_calcification = 4 

    Iter_mass = 10 
    Iter_calcification = 9  

    Label_mass = 'Mass' 
    Label_calcification = 'Calcification'  

    Mass_images_class = 0 
    Calcification_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_mass = dataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = True)
    Data_augmentation_calcification = dataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = Calcification_images_class, Saveimages = True)

    Images_mass, Labels_mass = Data_augmentation_mass.data_augmentation()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.data_augmentation()

    # * Add the value in the lists already created

    Images.append(Images_mass)
    Images.append(Images_calcification)

    Labels.append(Labels_mass)
    Labels.append(Labels_calcification)

    print(len(Images_mass))
    print(len(Images_calcification))

    return Images, Labels

def preprocessing_DataAugmentation_Biclass_CNN(Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 20 
    #Iter_Clasification = 40 

    #Iter_normal = 18 
    #Iter_Clasification = 34

    Iter_mass = 10 
    Iter_calcification = 9  

    Label_mass = 'Mass' 
    Label_calcification = 'Calcification'  

    Mass_images_class = 0 
    Clasification_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_mass = dataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = True)
    Data_augmentation_calcification = dataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = Clasification_images_class, Saveimages = True)

    Images_mass, Labels_mass = Data_augmentation_mass.data_augmentation_test_images()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_mass + Images_calcification
    Labels_total = np.concatenate((Labels_mass, Labels_calcification), axis = None)

    print(Images_mass)
    print(Images_calcification)

    #print(len(Images_mass))
    #print(len(Images_calcification))

    return Images_total, Labels_total
