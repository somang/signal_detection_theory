import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT
import numpy as np
import numpy.polynomial.polynomial as poly

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from cycler import cycler

from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit

import math
from math import exp,sqrt,pi
from line import Line
import re

def plot_roc_curve(fpr, tpr, var, ax, p=False): # takes false-positive rate, true-positive rate 
    if p:
        x = np.linspace(0, 1, 100)
        coefs = poly.polyfit(fpr, tpr, 2) # Polynomial fitting line, this can be used instead of the default line.
        ffit = poly.polyval(x, coefs)
        ax.plot(x, ffit, label='polyfit') # polyfit - fitting line with the dots.
    
    ax.plot(fpr, tpr, label='ROC', marker='.', linewidth=2,  markersize=10) #, linestyle="None")    
    ax.plot([0, 1], [0, 1], linestyle='--', label="d'=0", linewidth=2,  markersize=10) # guide line
    ax.set_xlabel('False-Alarm Rate', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title(var + "-" + 'ROC Curve')
    ax.legend(loc='best', prop={'size': 12})
    ax.tick_params(labelsize=14)
    return ax

def get_cumul_z(rate):
    tmp, sum = [], 0
    for i in range(len(rate)):
        sum += float(rate[i])
        tmp.append(sum)
    return tmp






# What do I need?
# input: factor variable value x
# output: probability of hit (H) and false-alarm (FA) given x

# what to do?
# convert x (ms, wpm, mw) to a z-score for the given 'signal' type.
# then get the c value probabilities p(H) and p(FA)

if __name__ == "__main__":
    #files = ['input/a_rating_data.in', 'input/d_rating_data.in', 'input/h_rating_data.in']
    files = [ 'input/d_confidence_rating_data.in', 'input/h_confidence_rating_data.in']
    for input_file in files:
        print("\nprocessing...................................................", input_file)
        with open(input_file) as f:
            """#######################################################
            ## GET FALSE-ALARM RATE FIRST, TO COMPARE WITH THE REST ##
            #######################################################"""
            content = f.readlines()
            fa_line = content[0]
            fa_list = list(map(lambda x: x.strip('\n'), fa_line.split("\t")))
            fa_list = fa_list[:1] + list(map(lambda x: float(x), fa_list[1:])) # conversion to float
            false_alarm, correct_rejection = sum(fa_list[1:6]), sum(fa_list[6:]) # get summary values for fa and cr
            for i in range(1, len(fa_list[1:])):
                fa_list[i] = fa_list[i] / (false_alarm + correct_rejection)
            
            """###############################################################
            ## GET HIT RATE, INTEGRATE THE CONFIDENCE RATINGS FOR ROC CURVE ##
            ###############################################################""" 
            for line in range(1, len(content)-1, 1):
                hit_line = content[line]
                hit_list = list(map(lambda x: x.strip('\n'), hit_line.split("\t")))
                v_name = hit_list[0]
                hit_list = hit_list[:1] + list(map(lambda x: float(x), hit_list[1:])) #convert to float
                hit, miss = sum(hit_list[1:6]), sum(hit_list[6:]) # get hit and miss
                
                sdt_obj = SDT(HI=hit, MI=miss, CR=correct_rejection, FA=false_alarm)
                #print(sdt_obj, "d'=" + str(sdt_obj.dprime()), "c=" + str(sdt_obj.c()))
                
                # Now lets create a list with the rates of hit and false alarm
                for i in range(1, len(hit_list[1:])):
                    hit_list[i] = hit_list[i] / (hit + miss)

                tpr, fpr = hit_list[1:], fa_list[1:]
                tpr_cum, fpr_cum = get_cumul_z(tpr), get_cumul_z(fpr)

                coefs = poly.polyfit(fpr, tpr, 2) # Polynomial fitting line, this can be used instead of the default line.
                print(v_name, coefs)
                # x = np.linspace(0, 1, 10)
                # ffit = poly.polyval(x, coefs)
                


#To create a frozen distribution:
fig, ax = plt.subplots(1, 1)

d_yns = {
    1:(12, 13), 2:(11, 14), 3:(9, 16), 4:(9, 16), 5:(8, 17), 6:(13, 12), 7:(12, 13), 8:(9, 16),
    9:(13, 12), 10:(12, 13), 11:(9, 16), 12:(9, 16), 13:(11, 14), 14:(15, 10), 15:(15, 10),
    16:(12, 13), 17:(8, 17), 18:(9, 16), 19:(12, 13), 20:(14,11), 21:(11,14), 22:(12,13)
}

hoh_yns = { 
    1:(8,19), 2:(7,20), 3:(8,19), 4:(10,17), 5:(9,18),6:(9,18),7:(4,23),8:(14,13),9:(9,18),10:(13,14),
    11:(11,16),12:(9,18),13:(12,15),14:(12,15),15:(14,13),16:(11,16),17:(8,19),18:(8,19),19:(10,17),
    20:(14,13),21:(9,18),22:(7,20)
}

# each rating pairs are:
# (rating when people said 'yes', rating when people said 'no')
# where 5 means high satisfactory quality and 1 means dissatisfacton
d_avg_ratings = {
    6: (2.000, 4.083), 9: (1.769, 3.833), 14: (2.133, 3.800), 15: (1.800, 3.900), 20: (1.929, 3.636)
}

hoh_avg_ratings = {
    4: (2.900, 3.765), 5: (2.111, 4.111), 6: (2.889, 3.944), 8: (2.929, 4.000), 9: (2.222, 3.833), 10: (2.154, 4.000),
    11: (1.909, 4.125), 12: (2.444, 4.222), 13: (2.000, 3.933), 14: (2.833, 4.133), 15: (2.286, 4.000), 16: (2.636, 4.125), 
    19: (1.700, 4.059), 20: (2.714, 3.923), 21: (2.444, 3.833)
}

yns = d_yns
v1 = yns[1]
for v in range(2, 23):
    h, m, fa, cr = yns[v][0], yns[v][1], v1[0], v1[1]
    sdt_obj = SDT(HI=h, MI=m, FA=fa, CR=cr)
    ph, pm, pfa, pcr = h/(h+m), m/(h+m), fa/(fa+cr), cr/(fa+cr)
    
    # generate
    noi_d = stats.norm(loc=0, scale=1)
    sig_d = stats.norm(loc=sdt_obj.dprime(), scale=1) #where loc is the mean and scale is the std dev
    
    # estimated rates
    epm = sig_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c()) # estimated_probability_of_miss
    epcr = noi_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c()) # estimated_probability_of_correctrejection
    eph = 1-epm # sig_d.sf(sdt_obj.c()) # estimated_probability_of_hit
    epfa = 1-epcr # noi_d.sf(sdt_obj.c()) #estimated_probability_of_falsealarm
""" 
    ####################################################################
    # let's find a function that can predict ratings from hit rates.
    #######################
    # deaf group first.
    if v in d_avg_ratings.keys():
        ptset_ph = [0, 1]
        ptset_y = [5, 0]
        # each point has a tuple (p(H), Rating).

        print("{}: p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} => R:{}".format(v, eph, epm, epfa, d_avg_ratings[v]))
        # now, let's add two more points.
        ptset_ph.append(eph)
        ptset_y.append(d_avg_ratings[v][0])
        ptset_ph.append(epm)
        ptset_y.append(d_avg_ratings[v][1])

        ######################### then, now we have four points ready to be fitted...
        ### first, one variable and linear polynomial regression (degree of 2)
        # ptset_x = ptset_ph
        # coefs = poly.polyfit(ptset_x, ptset_y, 2) # Fit with polyfit
        # print(coefs)
        # f = np.poly1d(coefs)
        # print(f)
        # x_new = np.linspace(0, 1, 100)
        # predicted_ratings = poly.polyval(x_new, coefs)
        # plt.plot(x_new, predicted_ratings)
        # plt.scatter(ptset_x, ptset_y, marker='.', color="red")
        # plt.ylabel('predicted_ratings')
        # plt.xlabel('hit_rates')
        # plt.show()

        ####################################################### what about p(FA) rates then?
        # Let's include pfa
        ptset_pfa = [epfa, epfa, epfa, epfa] # to match the number of points we have
        # now this becomes a multivariate regression model    
        #X is the independent variable (bivariate in this case)
        X = []
        for i in range(len(ptset_ph)):
            X.append([ptset_ph[i], ptset_pfa[i]])

        polynom_feat = PolynomialFeatures(degree=2)        #generate a model of polynomial features        
        X_ = polynom_feat.fit_transform(X) #transform the x data for proper fitting (for single variable type it returns, [1, x, x**2])
        #print(X_)

        reg = linear_model.LinearRegression() # generate the regression object
        #preform the actual regression
        reg.fit(X_, ptset_y) # ptset_y is the dependent data
        
        # now we have a fitted regression function.
        # let's try to predict... [p(H), p(FA)] -> R
        X_test = [
            [0.6, 0.48], 
            [0, epfa],
            [1, epfa]
        ]
        X_test_ = polynom_feat.fit_transform(X_test)

        # regression coefficients 
        #print('Coefficients: \n', reg.coef_) 
        print("X = \n", X_test_)
        
        Y_test = reg.predict(X_test_)
        
        for i in range(len(Y_test)):
            print("Predictions = {:.3f}".format(Y_test[i]))
 """