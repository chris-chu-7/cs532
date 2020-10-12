#CS 532 Assignment 4



from scipy.io import loadmat
import numpy as np



#Part A

#Least Squares Classifier Training. w = (X^T X)^(-1)X^T y
def find_weights():
    data = loadmat('face_emotion_data.mat')
    data_x_points = data['X']
    data_y_points = data['y']
    weights = np.linalg.inv(data_x_points.transpose()@data_x_points)@data_x_points.transpose()@data_y_points
    return weights

#find the result of the least squares classifier.
def find_y_value():
    data = loadmat('face_emotion_data.mat')
    data_x_points = data['X']
    data_y_points = data['y']
    weights = np.linalg.inv(data_x_points.transpose()@data_x_points)@data_x_points.transpose()@data_y_points
    return data_x_points@weights
    

print("\nWeights: ")
print(find_weights())

print("\nResults: ")
print(find_y_value())



#Part B

#There are 9 features in the corresponding feature vector to help decide on facial expression sentiments. The
#feature vector is represented the following way:

#feature = [x1 x2 x3 x4 x5 x6 x7 x8 x9]

#To find a solution to this particular problem, I must perform the following:

#result = x^T(w) 

#I then see if the result is positive or negative to perform the classification. If the result is positive, then
#the person is classified as "Happy". Otherwise, the person is classified as "angry".



#Part C

#Each column is normalized to scale. As a result, I strongly believe that feature 1 is the most important feature, since it
#has the most weight out of any feature and therefore has the greatest affect on wheather a person is smiling or not.



#Part D

#Classifier based on 3 of the 9 features. These features are feature 0, feature 2 and feature 3, since they contain the 
#greatest amounts of magnitude. I make a new matrix concatenating only 3 features instead of 9, same number of rows, with 
#these weights. Then, I multiply the values of this matrix together. 

#Part E

#Finding the percent of training labels incorrectly classified

classification_9 = find_y_value()
result_y = data['y']

i = 0
num_errors_9 = 0

for entry in result_y:
    if(entry > 0):
        if(classification_9[i] < 0):
            num_errors_9 = num_errors_9 + 1
        i = i + 1
    else:
        if(classification_9[i] > 0):
            num_errors_9 = num_errors_9 + 1
        i = i + 1
         
print("\nFor 9 features, " + str(num_errors_9) + " labels are incorrectly classified out of " + str(i) + ".")

#now I only want to take rows 0, 2, and 3. I do this by trying to extract/concatenate these rows and put them in a matrix
result_x = np.array(data['X']).transpose()
result_x_3 = np.array([result_x[0], result_x[2], result_x[3]])
result_weight = find_weights()
result_weight_3 = np.array([result_weight[0], result_weight[2], result_weight[3]])
result_mat_3 = np.matmul(np.array(result_x_3).transpose() , result_weight_3)
print(result_mat_3)

i = 0
num_errors_3 = 0

for entry in result_y:
    if(entry > 0):
        if(result_mat_3[i] < 0):
            num_errors_3 = num_errors_3 + 1
        i = i + 1
    else:
        if(result_mat_3[i] > 0):
            num_errors_3 = num_errors_9 + 1
        i = i + 1

print("\nFor 3 features, " + str(num_errors_3) + " labels are incorrectly classified out of " + str(i) + ".")

#This means that 2.34% training labels are incorrecly classified using 9 features and 3.12% 
#of labels are incorrectly classified using 3 features. 




