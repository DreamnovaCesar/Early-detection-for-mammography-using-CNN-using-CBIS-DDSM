
import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import splitfolders
import tensorflow as tf

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ? Detect fi GPU exist in your PC for CNN

def detect_GPU():

  # * This function shows if a gpu device is available and its name. 
  # * this is good to know if the training is using a GPU 

  GPU_name = tf.test.gpu_device_name()
  GPU_available = tf.test.is_gpu_available()

  #print(GPU_available)

  if GPU_available == True:
      print("GPU device is available")

  if "GPU" not in GPU_name:
      print("GPU device not found")
  print('Found GPU at: {}'.format(GPU_name))

# ? Sort Files

def sort_images(Folder_path): 

	"""
	Read all images in a folder and sort them.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	  int:Returning value
    int:Returning list[str]

   	"""

  # * This function sort the files and show them

	Number_images = len(os.listdir(Folder_path))

	print("\n")
	print("********************************")
	print(f"Images: {Number_images}")
	print("********************************")
	print("\n")

	files = os.listdir(Folder_path)
	print(files)
	print("\n")

	print("********************************")
	Sorted_files =  sorted(files)
	print(Sorted_files)
	print("\n")
	print("********************************")

	return Sorted_files, Number_images

# ? Remove all files in folder

def remove_all_files(Folder_path):

	"""
	Remove all images inside the folder chosen

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	Void

   	"""
  # * This function will remove all the files inside a folder

	for File in os.listdir(Folder_path):
		filename, extension  = os.path.splitext(File)
		print(f"Removing {filename} {extension}✅")
		os.remove(os.path.join(Folder_path, File))

def CBIS_DDSM_split_data(Folder_CSV, Folder, Folder_total_benign, Folder_benign, Folder_benign_wc, Folder_malignant, Folder_abnormal, Dataframe, Severity, Phase):

    remove_all_files(Folder_total_benign)
    remove_all_files(Folder_benign)
    remove_all_files(Folder_benign_wc)
    remove_all_files(Folder_malignant)
    remove_all_files(Folder_abnormal)

    Images = []
    Label = []

    Filename_benign_all = []
    Filename_malignant_all = []
    Filename_all = []

    Filename_benign_list = []
    Filename_benign_WC_list = []
    Filename_malignant_list = []
    Filename_abnormal_list = []

    Benign = 0
    Benign_with_callback = 1
    Malignant = 2

    Index = 0
        
    os.chdir(Folder)

    Sorted_files, Total_images = sort_images(Folder)
    Index = 0
    Count = 1

    for File in Sorted_files:
        
        Filename, Format = os.path.splitext(File)

        if Dataframe[Index] == Benign:

                try:

                    print(f"Working with {Count} of {Total_images} Benign images, {Filename}")
                    Count += 1

                    File_folder = os.path.join(Folder, File)
                    Image_benign = cv2.imread(File_folder)

                    Filename_benign = Filename + '_Benign'
                    Filename_benign_format = Filename_benign + Format

                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign)

                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_benign_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_benign)

                    Filename_benign_folder = os.path.join(Folder_benign, Filename_benign_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign)

                    Images.append(Image_benign)
                    Label.append(Benign)

                    Filename_benign_all.append(Filename_benign)
                    Filename_abnormal_list.append(Filename_benign)
                    Filename_benign_list.append(Filename_benign)
                    Filename_all.append(Filename_benign)

                except OSError:
                    print('Cannot convert %s' % File)

        elif Dataframe[Index] == Benign_with_callback:
    
                try:

                    print(f"Working with {Count} of {Total_images} Benign_without_callback images, {Filename}")
                    Count += 1

                    File_folder = os.path.join(Folder, File)
                    Image_benign_without_callback = cv2.imread(File_folder)

                    Filename_benign_WC = Filename + '_Benign_Without_Callback'
                    Filename_benign_WC_format = Filename_benign_WC + Format

                    Filename_total_benign_folder = os.path.join(Folder_total_benign, Filename_benign_WC_format)
                    cv2.imwrite(Filename_total_benign_folder, Image_benign_without_callback)

                    Filename_benign_folder = os.path.join(Folder_benign_wc, Filename_benign_WC_format)
                    cv2.imwrite(Filename_benign_folder, Image_benign_without_callback)

                    Images.append(Image_benign_without_callback)
                    Label.append(Benign_with_callback)

                    Filename_benign_all.append(Filename_benign_WC)
                    Filename_benign_WC_list.append(Filename_benign_WC)
                    Filename_all.append(Filename_benign_WC)

                except OSError:
                    print('Cannot convert %s' % File)
        
        elif Dataframe[Index] == Malignant:

                try:

                    print(f"Working with {Count} of {Total_images} Malignant images, {Filename}")
                    Count += 1

                    File_folder = os.path.join(Folder, File)
                    Image_malignant = cv2.imread(File_folder)

                    Filename_malignant = Filename + '_Malignant'
                    Filename_malignant_format = Filename_malignant + Format

                    Filename_abnormal_folder = os.path.join(Folder_abnormal, Filename_malignant_format)
                    cv2.imwrite(Filename_abnormal_folder, Image_malignant)

                    Filename_malignant_folder = os.path.join(Folder_malignant, Filename_malignant_format)
                    cv2.imwrite(Filename_malignant_folder, Image_malignant)

                    Images.append(Image_malignant)
                    Label.append(Malignant)

                    Filename_malignant_all.append(Filename_malignant)
                    Filename_abnormal_list.append(Filename_malignant)
                    Filename_malignant_list.append(Filename_malignant)
                    Filename_all.append(Filename_malignant)

                except OSError:
                    print('Can not convert %s' % File)
                    
        Index += 1

    Dataframe_all = pd.DataFrame({'Filename_All':Filename_all,'Labels':Label}) 

    Dataframe_all_name = 'CBIS_DDSM_Split_' + 'Dataframe' + str(Severity) + '_' + str(Phase) + '.csv' 
    Dataframe_all_name_folder = os.path.join(Folder_CSV, Dataframe_all_name)

    Dataframe_all.to_csv(Dataframe_all_name_folder)

    return Dataframe_all

def CBIS_DDSM_split_several_data(Folder_Path_Test, Folder_Path_Training, Folder_Path_Benign_Total, Folder_Path_Benign, Folder_Path_BenignWC, Folder_Path_Malignant, Folder_Path_Abnormal, Dataset_Test, Dataset_Training):

  remove_all_files(Folder_Path_Benign_Total)
  remove_all_files(Folder_Path_Benign)
  remove_all_files(Folder_Path_BenignWC)
  remove_all_files(Folder_Path_Malignant)
  remove_all_files(Folder_Path_Abnormal)

  Images = []
  Label = []

  Filename_Benign_All = []
  Filename_Malignant_All = []
  Filename_All = []

  Filename_Benign = []
  Filename_Benign_WC = []
  Filename_Malignant = []
  Filename_Abnormal = []

  Benign = 0
  Benign_with_callback = 1
  Malignant = 2

  png = ".png"    # png.

  os.chdir(Folder_Path_Test)

  sorted_files, images = sort_images(Folder_Path_Test)
  Index = 0
  count = 1
  
  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)
    
    if Dataset_Test[Index] == Benign:

        try:

            print(f"Working with {count} of {images} Benign images, {filename}")
            count += 1

            Path_File = os.path.join(Folder_Path_Test, File)
            Imagen_Benign = cv2.imread(Path_File)

            FileName_Benign = filename + '_benign'
            dst_name = FileName_Benign + png

            dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            #Images.append(Imagen_Benign)
            Label.append(Benign)

            Filename_Benign_All.append(FileName_Benign)
            Filename_Benign.append(FileName_Benign)
            Filename_All.append(FileName_Benign)

        except OSError:
            print('Cannot convert %s' % File)

    elif Dataset_Test[Index] == Benign_with_callback:
  

            try:

                print(f"Working with {count} of {images} Benign_without_callback images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Test, File)
                Imagen_Benign_without_callback = cv2.imread(Path_File)

                FileName_Benign_WC = filename + '_benign_without_callback'
                dst_name = FileName_Benign_WC + png

                dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                #cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                dstPath_name = os.path.join(Folder_Path_BenignWC, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #Images.append(Imagen_Benign_without_callback)
                Label.append(Benign_with_callback)

                Filename_Benign_All.append(FileName_Benign_WC)
                Filename_Benign_WC.append(FileName_Benign_WC)
                Filename_All.append(FileName_Benign_WC)

            except OSError:
                print('Cannot convert %s' % File)
      
    elif Dataset_Test[Index] == Malignant:


            try:

                print(f"Working with {count} of {images} Malignant images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Test, File)
                Imagen_Malignant = cv2.imread(Path_File)

                FileName_Malignant = filename + '_malignant'
                dst_name = FileName_Malignant + png

                dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                dstPath_name = os.path.join(Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                #Images.append(Imagen_Malignant)
                Label.append(Malignant)

                Filename_Malignant_All.append(FileName_Malignant)
                Filename_Malignant.append(FileName_Malignant)
                Filename_All.append(FileName_Malignant)

            except OSError:
                print('Cannot convert %s' % File)

    Index += 1
    print(Index)
    
  os.chdir(Folder_Path_Training)

  sorted_files, images = ShowSort(Folder_Path_Training)
  Index = 0
  count = 1

  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)

    if Dataset_Training[Index] == Benign:

        try:

            print(f"Working with {count} of {images} Benign images, {filename}")
            count += 1

            Path_File = os.path.join(Folder_Path_Training, File)
            Imagen_Benign = cv2.imread(Path_File)

            FileName_Benign = filename + '_benign'
            dst_name = FileName_Benign + png

            dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            #Images.append(Imagen_Benign)
            Label.append(Benign)

            Filename_Benign_All.append(FileName_Benign)
            Filename_Benign.append(FileName_Benign)
            Filename_All.append(FileName_Benign)

        except OSError:
            print('Cannot convert %s' % File)

    elif Dataset_Training[Index] == Benign_with_callback:
  

            try:

                print(f"Working with {count} of {images} Benign_without_callback images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Training, File)
                Imagen_Benign_without_callback = cv2.imread(Path_File)

                FileName_Benign_WC = filename + '_benign_without_callback'
                dst_name = FileName_Benign_WC + png

                dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                dstPath_name = os.path.join(Folder_Path_BenignWC, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #Images.append(Imagen_Benign_without_callback)
                Label.append(Benign_with_callback)

                Filename_Benign_All.append(FileName_Benign_WC)
                Filename_Benign_WC.append(FileName_Benign_WC)
                Filename_All.append(FileName_Benign_WC)

            except OSError:
                print('Cannot convert %s' % File)
      
    elif Dataset_Training[Index] == Malignant:


            try:

                print(f"Working with {count} of {images} Malignant images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Training, File)
                Imagen_Malignant = cv2.imread(Path_File)

                FileName_Malignant = filename + '_malignant'
                dst_name = FileName_Malignant + png

                dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                dstPath_name = os.path.join(Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                #Images.append(Imagen_Malignant)
                Label.append(Malignant)

                Filename_Malignant_All.append(FileName_Malignant)
                Filename_Malignant.append(FileName_Malignant)
                Filename_All.append(FileName_Malignant)

            except OSError:
                print('Cannot convert %s' % File)
                
    Index += 1

  Dataset = pd.DataFrame({'Filename_All':Filename_All,'Labels':Label}) 

  return Images, Dataset

def DatasetSplit(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Dataset['Labels'].value_counts()

  return X, Y, Majority, Minority

def ImbalanceDataMajority(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Y.value_counts()

  df_majority = Dataset[Y == 0]
  df_minority = Dataset[Y == 1]

  df_majority_downsampled = resample( df_majority, 
                                      replace = False,     # sample with replacement
                                      n_samples = Minority, # to match majority class
                                      random_state = 123) # reproducible results
  
  df_downsampled = pd.concat([df_minority, df_majority_downsampled])
  print(df_downsampled['Labels'].value_counts())

  X = df_downsampled.drop('Labels', axis = 1)
  Y = df_downsampled['Labels']

  return X, Y, Majority, Minority

def ImbalanceDataMinority(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Y.value_counts()

  df_majority = Dataset[Y == 0]
  df_minority = Dataset[Y == 1]

  df_minority_upsampled = resample( df_minority, 
                                    replace = True,     # sample with replacement
                                    n_samples = Majority, # to match majority class
                                    random_state = 123) # reproducible results

  df_upsampled = pd.concat([df_majority, df_minority_upsampled])
  print(df_upsampled['Labels'].value_counts())

  X = df_upsampled.drop('Labels', axis = 1)
  Y = df_upsampled['Labels']

  return X, Y, Majority, Minority

# Biclass printing data augmentation

def BiclassPrinting(ImagesNormal, ImagesTumor, Technique):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Tumor images.
    argument3 (str): Technique used

    Returns:
	  void
   	"""

    print("\n")
    print(Technique + ' images: ' + str(len(ImagesNormal)))
    print(Technique + ' images: ' + str(len(ImagesTumor)))

# Triclass printing data augmentation

def TriclassPrinting(ImagesNormal, ImagesBenign, ImagesMalignant, Technique):

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

    print("\n")
    print(Technique + ' images: ' + str(len(ImagesNormal)))
    print(Technique + ' images: ' + str(len(ImagesBenign)))
    print(Technique + ' images: ' + str(len(ImagesMalignant)))

# Convertion severity to int value

def CBIS_DDSM_CSV_severity_labeled(Folder_CSV, Column, Severity):

    LE = LabelEncoder()
    Calcification = 1
    Mass = 2

    if Severity == Calcification:
        Columns_list = ["patient_id", "breast density", "left or right breast", "image view", "abnormality id", "abnormality type", "calc type", "calc distribution", "assessment", "pathology", "subtlety", "image file path", "cropped image file path", "ROI mask file path"]
    elif Severity == Mass:
        Columns_list = ["patient_id", "breast_density", "left or right breast", "image view", "abnormality id", "abnormality type", "mass shape", "mass margins", "assessment", "pathology", "subtlety", "image file path", "cropped image file path", "ROI mask file path"]
    
    Dataframe_severity = pd.read_csv(Folder_CSV, usecols = Columns_list)

    Dataframe_severity.iloc[:, Column].values
    Dataframe_severity.iloc[:, Column] = LE.fit_transform(Dataframe_severity.iloc[:, Column])

    Dataset_severity_labeled = Dataframe_severity.iloc[:, Column].values
    Dataframe = Dataframe_severity.iloc[:, Column]

    print(Dataset_severity_labeled)
    pd.set_option('display.max_rows', Dataframe.shape[0] + 1)
    print(Dataframe.value_counts())

    return Dataset_severity_labeled

#

def concat_dataframe(*dfs, **kwargs):

    # * this function concatenate the number of dataframes added

    # * General parameters

    Folder = kwargs.get('Folder', None)
    Class_problem = kwargs.get('Class', None)
    Technique = kwargs.get('Technique', None)
    Save_CSV = kwargs.get('SaveCSV', False)

    if Folder == None:
      raise ValueError("Folder does not exist") #! Alert

    elif Class_problem == None:
      raise ValueError("Class does not exist")  #! Alert

    elif Technique == None:
      raise ValueError("Technique does not exist")  #! Alert

    # * Concatenate each dataframe
    ALL_dataframes = [df for df in dfs]

    Final_dataframe = pd.concat(ALL_dataframes, ignore_index = True, sort = False)
        
    #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
    #print(DataFrame)

    if Save_CSV == True:

        # * Name the final dataframe and save it into the given path
        Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
        Folder_dataframe_to_save = os.path.join(Folder, Name_dataframe)
        Final_dataframe.to_csv(Folder_dataframe_to_save)

    return Final_dataframe

# ? Split folders into train/test/validation

def split_folders_train_test_val(Folder):

    #Name_dir = os.path.dirname(Folder)
    #Name_base = os.path.basename(Folder)

    Name_base_mod = Folder + '_Split'

    splitfolders.ratio(Folder, output = Name_base_mod, seed = 1337, ratio = (0.8, 0.1, 0.1)) 

    print(Name_base_mod)
    """
    for (root, dirs, files) in os.walk(Name_base_mod, topdown = True):
        print (root)
        print (dirs)
        #print (files)
        print ('--------------------------------')
    """

def ConfigurationModels(MainKeys, Arguments, MultiDataModels, MultiDataModelsEsp):

    TotalImage = []
    TotalLabel = []

    ClassSize = (len(Arguments[2]))
    Images = 7
    Labels = 8

    if len(Arguments) == len(MainKeys):
        
        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            #print('Total:', len(TotalImage))
        
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            #print('Total:', len(TotalLabel))

            Images += 2
            Labels += 2

        #TotalImage = [*list(DicAruments.values())[Images], *list(DicAruments.values())[Images + 2]]
        
    elif len(Arguments) > len(MainKeys):

        TotalArguments = len(Arguments) - len(MainKeys)

        for i in range(TotalArguments // 2):

            MainKeys.append('Images ' + str(i + 3))
            MainKeys.append('Labels ' + str(i + 3))

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            Images += 2
            Labels += 2

    elif len(Arguments) < len(MainKeys):

        raise ValueError('No se puede xD')

    #print(DicAruments)

    def printDict(DicAruments):

        for i in range(7):
            print(list(DicAruments.items())[i])

    printDict(DicAruments)

    print(len(TotalImage))
    print(len(TotalLabel))

    X_train, X_test, y_train, y_test = train_test_split(np.array(TotalImage), np.array(TotalLabel), test_size = 0.20, random_state = 3, shuffle = True)

    Score = PreTrainedModels(Arguments[0], Arguments[1], Arguments[2], Arguments[3], Arguments[4], ClassSize, Arguments[5], Arguments[6], X_train, y_train, X_test, y_test, MultiDataModels, MultiDataModelsEsp)
    #Score = PreTrainedModels(ModelPreTrained, technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test)
    return Score

#

def UpdateCSV(Score, df, column_names, path, row):

    print(df.head(len(df.index)))

    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
    
    print(df.head(len(df.index)))

    df.to_csv(path, index = False)
  
    print(df)

# Concat multiple dataframes

def DataframeSave(*dfs, **kwargs):

    folder = kwargs.get('folder', None)
    Class = kwargs.get('Class', None)
    technique = kwargs.get('technique', None)

    if folder == None:
      raise ValueError("Folder does not exist")

    elif Class == None:
      raise ValueError("Class does not exist")

    elif technique == None:
      raise ValueError("Technique does not exist")

    ALLdf = [df for df in dfs]

    DataFrame = pd.concat(ALLdf, ignore_index = True, sort = False)
        
    pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    #print(DataFrame)

    dst =  str(Class) + '_Dataframe_' + str(technique) + '.csv'
    dstPath = os.path.join(folder, dst)

    DataFrame.to_csv(dstPath)