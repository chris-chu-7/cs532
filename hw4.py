#CS 532 Assignment 4

from scipy.io import loadmat
import numpy as np

#Part A


#Least Squares Classifier Training. w = (X^T X)^(-1)X^T y
def findweights():
    data = loadmat('face_emotion_data.mat')
    data_x_points = data['X']
    data_y_points = data['y']
    weights = np.linalg.inv(data_x_points.transpose()@data_x_points)@data_x_points.transpose()@data_y_points
    return weights

print(findweights())



#Part B


#There are 9 features in the corresponding feature vector to help decide on facial expression sentiments. The
#feature vector is represented the following way:

#feature = [x1 x2 x3 x4 x5 x6 x7 x8 x9]

#To find a solution to this particular problem, I must perform the following:

#result = x^T(w) 

#I then see if the result is positive or negative to perform the classification. If the result is positive, then
#the person is classified as "Happy". Otherwise, the person is classified as "angry".





