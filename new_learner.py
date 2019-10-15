from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from scipy import stats
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

import numpy as np
from math import sqrt

EPOCHS = 50  # Number of reps
BATCH = 500
DATAFILE = "c_user_model_data100000.csv"
TRAINING = 1

def data_prep():
    ###########LOAD DATA ##############
    dataset = loadtxt(DATAFILE, delimiter=',')
    print(len(dataset))
    t_index = round(len(dataset) * 0.8)  # 90% to train
    # split into input (X) and output (y) variables
    np.set_printoptions(precision=4, suppress=True)

    sc = StandardScaler()
    x = sc.fit_transform(dataset[:, :5])
    # Range 0:4 selects columns from 0 to 3, stopping before index 4

    # Just spits out first row.
    X = dataset[0, 0:5]
    Y = dataset[0, 5:]
    print (X, Y)
    x_tr = x[:t_index, :5]
    x_ts = x[t_index:, :5]
    y_tr = dataset[:t_index, 5:]  # regression for training
    y_ts = dataset[t_index:, 5:]  # relu for testing

    return (x_tr, x_ts, y_tr, y_ts)

# define keras model
def train_nn_reglu(x_tr, x_ts, y_tr, y_ts):
    model = Sequential()
    model.add(Dense(units=6, input_dim=5, activation='relu'))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=8, activation='softmax'))
    model.add(Dense(units=4))

    # compile keras model
    #model.compile(loss='mean_absolute_error', optimizer='RMSprop', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])

    # Fit Keras Model
    hist = model.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
              verbose=1, validation_data=(x_ts, y_ts))

    return hist, model

def print_model_perf(predictions, x_ts, y_ts, name):
    category_set = ["Delay", "Speed", "Missing Words", "Paraphrasing"]
    col_set = ['g', 'b', 'y', 'c']
    for i in range(4):
        color = col_set[i]
        category = category_set[i]
        rms = sqrt(mean_squared_error(y_ts[:, i], predictions[:, i]))
        print(name + " " + category_set[i] + "\nRMSE: {:.2f}".format(rms))
        correct = 0
        rounded_p = np.rint(predictions)
        for j in range(len(rounded_p)):
            if rounded_p[j, i] == y_ts[j, i]:
                correct += 1
        # print(correct,"correct answers out of",len(rounded_p))
        print("acc: {:.2f}%".format(correct / len(rounded_p[:, i]) * 100.0))
    print()

if __name__ == '__main__':
    print(DATAFILE)
    if TRAINING:
        (x_tr, x_ts, y_tr, y_ts) = data_prep()
        hist, model = train_nn_reglu(x_tr, x_ts, y_tr, y_ts)


    #Graph Prediction VS Real
    predictions = model.predict(x_ts, batch_size=500)
    print_model_perf(predictions, x_ts, y_ts, "Multilayer Perceptron:")

    # Linear Regression
    # mlm = linear_model.LinearRegression()
    # stat_model = mlm.fit(x_tr, y_tr)
    # predictions = mlm.predict(x_ts)
    # print_model_perf(predictions, x_ts, y_ts, "MLM")
    
    ##### Polynomial linear regression
    poly = PolynomialFeatures(degree=3)
    training_x = poly.fit_transform(x_tr)
    testing_x = poly.fit_transform(x_ts)
    lg = linear_model.LinearRegression()
    lg.fit(training_x, y_tr)
    predictions = lg.predict(testing_x)
    print_model_perf(predictions, x_ts, y_ts, "Polynomial Regression:")
    
