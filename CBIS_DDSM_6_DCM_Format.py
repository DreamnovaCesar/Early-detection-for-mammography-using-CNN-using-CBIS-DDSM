
import os
import cv2
from cv2 import phase
import pydicom
import numpy as np
import pandas as pd

from CBIS_DDSM_2_General_Functions import sort_images
from CBIS_DDSM_2_General_Functions import remove_all_files

# ? class for images cropping.

class DCM_format():

  def __init__(self, **kwargs):
    
    # * This algorithm outputs crop values for images based on the coordinates of the CSV file.
    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.Folder_Data = kwargs.get('Datafolder', None)
    self.Folder_normal = kwargs.get('Normalfolder', None)
    self.Folder_resize = kwargs.get('Resizefolder', None)
    self.Folder_resize_normalize = kwargs.get('Normalizefolder', None)

    self.Severity = kwargs.get('Severity', None)
    self.Phase = kwargs.get('Phase', None)

    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif self.Folder_normal == None:
      raise ValueError("Folder for normal images does not exist") #! Alert

    elif self.Folder_resize == None:
      raise ValueError("Folder for tumor images does not exist") #! Alert

    elif self.Folder_resize_normalize == None:
      raise ValueError("Folder for benign images does not exist") #! Alert

  def DCM_change_format(self):

    """
        Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Benign images.
    argument3 (list): The number of Malignant images.
    argument4 (str): Technique used

    Returns:
        void
    """

    X_size_resize = 224
    Y_size_resize = 224

    DCM = ".dcm"
    DCM_files = []
    DCM_files_sizes = []
    DCM_Filenames = []

    arr = os.listdir(self.Folder)

    Arr = arr * 2
    Arr = sorted(Arr)

    for Root, Dirs, Files in os.walk(self.Folder, True):
        print("root:%s"% Root)
        print("dirs:%s"% Dirs)
        print("files:%s"% Files)
        print("-------------------------------")

    for Root, Dirs, Files in os.walk(self.Folder):
        for x in Files:
            if x.endswith(DCM):
                DCM_files.append(os.path.join(Root, x))

    DCM_files = sorted(DCM_files)
    
    for i in range(len(DCM_files)):
        DCM_files_sizes.append(os.path.getsize(DCM_files[i]))

    DCM_dataframe_files = pd.DataFrame({'Path':DCM_files, 'Size':DCM_files_sizes, 'Filename':Arr}) 

    print(DCM_dataframe_files)

    Total = len(DCM_files_sizes)

    for i in range(0, Total, 2):

        print(DCM_files_sizes[i], '----', DCM_files_sizes[i + 1])

        if DCM_files_sizes[i] > DCM_files_sizes[i + 1]:
            DCM_dataframe_files.drop([i], axis = 0, inplace = True)
        else:
            DCM_dataframe_files.drop([i + 1], axis = 0, inplace = True)

    print(len(DCM_files))
    print(len(DCM_files_sizes))
    print(len(Arr))

    print(DCM_dataframe_files)

    DCM_Filenames = DCM_dataframe_files.iloc[:, 0].values
    Arr = DCM_dataframe_files.iloc[:, 2].values

    Interpolation = cv2.INTER_CUBIC

    Shape_resize = (X_size_resize, Y_size_resize)

    DCM_dataframe_name = 'DCM_' + 'Format_' + str(self.Severity) + '_' + str(self.Phase) + '.csv'
    DCM_dataframe_folder = os.path.join(self.Folder_Data, DCM_dataframe_name)
    DCM_dataframe_files.to_csv(DCM_dataframe_folder)

    File = 0

    for File in range(len(DCM_dataframe_files)):

        DCM_read_pydicom_file = pydicom.dcmread(DCM_Filenames[File])
        
        DCM_image = DCM_read_pydicom_file.pixel_array.astype(float)

        DCM_image_rescaled = (np.maximum(DCM_image, 0) / DCM_image.max()) * 255.0
        DCM_image_rescaled_float64 = np.float64(DCM_image_rescaled)

        DCM_black_image = np.zeros((X_size_resize, Y_size_resize))

        DCM_image_resize = cv2.resize(DCM_image_rescaled_float64, Shape_resize, interpolation = Interpolation)
        DCM_image_normalize = cv2.normalize(DCM_image_resize, DCM_black_image, 0, 255, cv2.NORM_MINMAX)

        DCM_file = Arr[File]

        DCM_name_file = str(DCM_file) + '.png'

        DCM_folder = os.path.join(self.Folder_normal, DCM_name_file)
        DCM_folder_resize = os.path.join(self.Folder_resize, DCM_name_file)
        DCM_folder_normalize = os.path.join(self.Folder_resize_normalize, DCM_name_file)

        cv2.imwrite(DCM_folder, DCM_image_rescaled_float64)
        cv2.imwrite(DCM_folder_resize, DCM_image_resize)
        cv2.imwrite(DCM_folder_normalize, DCM_image_normalize)

        print('Images: ', DCM_Filenames[File], '------', Arr[File])
 
