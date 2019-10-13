from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# load the dataset
dataset = loadtxt('c_user_model_data100000.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[0,0:4]
Y = dataset[0,4:]
print(X, Y)