import numpy as np

from CBIS_DDSM_4_Data_Augmentation import dataAugmentation

def preprocessing_DataAugmentation_Multiclass_ML(Folder_normal, Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters

    Iter_normal = 25
    Iter_mass = 4
    Iter_calcification = 3 

    #Iter_normal = 2
    #Iter_mass = 8
    #Iter_calcification = 10 

    Label_normal = 'Normal'
    Label_mass = 'Mass' 
    Label_calcification = 'Calcification'

    Normal_images_class = 0
    Mass_images_class = 1
    calcification_images_class = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, NewFolder = Folder_destination, Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_mass = dataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = True)
    Data_augmentation_calcification = dataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = calcification_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation()
    Images_mass, Labels_mass = Data_augmentation_mass.data_augmentation()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.data_augmentation()

    # * Add the value in the lists already created

    Images.append(Images_Normal)
    Images.append(Images_mass)
    Images.append(Images_calcification)

    Labels.append(Labels_Normal)
    Labels.append(Labels_mass)
    Labels.append(Labels_calcification)
    
    print(len(Images_Normal))
    print(len(Images_mass))
    print(len(Images_calcification))

    return Images, Labels

def preprocessing_DataAugmentation_Multiclass_CNN(Folder_normal, Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 2
    #Iter_benign = 70
    #Iter_calcification = 90 

    Iter_normal = 25
    Iter_mass = 4
    Iter_calcification = 3 

    #Iter_normal = 2
    #Iter_mass = 8
    #Iter_calcification = 10 

    Label_normal = 'Normal'
    Label_mass = 'Mass'
    Label_calcification = 'Clasification' 

    Normal_images_class = 0
    Mass_images_class = 1
    Calcification_images_class = 2 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_normal = dataAugmentation(Folder = Folder_normal, NewFolder = Folder_destination, Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_mass = dataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = True)
    Data_augmentation_calcification = dataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = Calcification_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_test_images()
    Images_mass, Labels_mass = Data_augmentation_mass.data_augmentation_test_images()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_mass + Images_calcification
    Labels_total = np.concatenate((Labels_Normal, Labels_mass, Labels_calcification), axis = None)
    
    print(Images_Normal)
    print(Images_mass)
    print(Images_calcification)

    #print(len(Images_normal))
    #print(len(Images_mass))
    #print(len(Images_calcification))

    return Images_total, Labels_total
