# Supervised Outlier Detection (Python 3) #
In this problem, images are provided of two categories – inlier and outliers. The inlier images heavily overshadow the outlier images. Basically, inlier images are those which have some text in their main body. The aim of this solution is to efficiently classify the test data images as outliers (1) and inliers (0).

## 1. Getting Started ##
The provided images are heavily imbalanced as the number of inliers are much higher than the outliers. Hence data augmentation is required. Here the images are changed with respect to dimensionality, zooming, rotation, etc.

### 1.1. Data Description ###
[Three folders](https://drive.google.com/drive/folders/1O5s-Xz9ChRFDsxhtI6WwtIff1LdGJnQK) are given (inlier_train, outlier_train and test) which contain all the images. Outlier folder contains about 800 images, inlier folder contains about 4000 images while the test folder has about 1300 images.

### 1.2. Prerequisites ###
The following libraries are required to run the code. All of them can be downloaded via conda or pip:
1. pandas
2. numpy
3. os, sys, cv2
4. keras
5. sklearn
6. PIL
7. shutil
Add all the three given folders into the same directory where the code is located.

### 1.3. Running the Code ###
The code can be run Jupyter Notebook.

## 2. Methodology ##
The following steps are followed to find the outliers in the given test image folder:
1. Import all the necessary libraries
2. Read all the filenames in inlier train and outlier train
3. Make two folders for storing the augmented images and the final train images
4. For each outlier image, create augmented images and save them in the augmented folder.
5. Shift all the images within the augmented folder into train folder. Also do the same for the outlier folder.
6. Read all the images within the train folder (augmented and outliers) and give them a label of 1.
7. Add a label of 0 to the inlier filenames and then shift the images to train folder as well.
8. Extract all features from training data images with ResNet50 model.
9. Do the same for test data images.
10. Scale the features of both test and train features with Standard Scaler
11. Train the KNN Classifier (classes =2) with the scaled train features and the labels (0,1).
12. Test the data and predict the labels for the test images.
13. Write the result in CSV in necessary format.

For this question I have used KNN Classifier as the model. SVC gave very poor number of outliers (1). Though OneClassSVM is usually used for such datasets, it was giving very low number of outliers for this particular dataset. Hence, I did not use it.

## 3. Author ##
BANERJEE, Rohini - HKUST MSc BDT (Student ID: 20543577)

## 4. References ##
1. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
2. https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
