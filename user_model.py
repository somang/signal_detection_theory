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
from scipy.stats import truncnorm
from scipy.optimize import curve_fit

import math
from math import exp,sqrt,pi
from line import Line
import re

import csv
import numpy as np
import matplotlib.pyplot as plt

from random import randint
from random import gauss


# What do I need?
# input: factor variable value x
# output: probability of hit (H) and false-alarm (FA) given x

# what to do?
# convert x (ms, wpm, mw) to a z-score for the given 'signal' type.
# then get the c value probabilities p(H) and p(FA)
"""
    Input: Delay, Speed, Number of Missing words, Words frequency, Paraphrasing

    Output: Probability of hit, Quality rating
    1. Use functions to generate regression function. What does it take as an input?
    2. Get mapping between standard scores to raw values
    3. Then, raw values will be taken as input, to return quality ratings.


    Ranges?
        Delay: 0 sec -> 10 sec?
        Speed: 0 WPM -> 160 -> 300 WPM (freq analysis)?
        # mw : freq analysis
        Paraphrase: boolean
"""


# from https://nessy.info/?p=16
# modified to fit sdt axis mapping

class Usermodels(object):
    d_yns = {1:(12,13),2:(11,14),3:(9,16),4:(9,16),5:(8,17),6:(13,12),7:(12,13),8:(9,16),9:(13,12), 
        10:(12, 13),11:(9,16),12:(9,16),13:(11,14),14:(15,10),15:(15,10),16:(12,13),17:(8,17),18:(9,16),19:(12,13),
        20:(14,11),21:(11,14),22:(12,13)
        }
    hoh_yns = {1:(8,19),2:(7,20),3:(8,19),4:(10,17),5:(9,18),6:(9,18),7:(4,23),8:(14,13),9:(9,18),
        10:(13,14),11:(11,16),12:(9,18),13:(12,15),14:(12,15),15:(14,13),16:(11,16),17:(8,19),18:(8,19),19:(10,17),
        20:(14,13),21:(9,18),22:(7,20)
        }
    d_avg_ratings = {6:(2.000, 4.083), 9:(1.769, 3.833), 14:(2.133, 3.800), 15:(1.800, 3.900), 20:(1.929, 3.636)
        }
    hoh_avg_ratings = {4:(2.900, 3.765), 5:(2.111, 4.111), 6:(2.889, 3.944), 8:(2.929, 4.000), 9:(2.222, 3.833), 10:(2.154, 4.000),
        11:(1.909, 4.125), 12:(2.444, 4.222), 13:(2.000, 3.933), 14:(2.833, 4.133), 15:(2.286, 4.000), 16:(2.636, 4.125), 
        19:(1.700, 4.059), 20:(2.714, 3.923), 21:(2.444, 3.833)
    }

    d_avg_ratings_all = {1:(2.250, 3.846), 2:(1.818, 4.071), 3:(1.889, 3.688), 4:(2.333, 4.000), 5:(1.750, 3.529),
        6:(2.000, 4.083), 7:(2.333, 3.923), 8:(2.333, 4.062), 9:(1.769, 3.833), 10:(2.000, 3.077),
        11:(1.667, 3.625), 12:(1.778, 3.438), 13:(2.091, 4.071), 14:(2.133, 3.800), 15:(1.800, 3.900), 
        16:(2.167, 3.846), 17:(2.250, 3.588), 18:(2.667, 3.938), 19:(2.083, 3.769), 20:(1.929, 3.636),
        21:(2.182, 3.714), 22:(1.917, 4.154)
        }
    hoh_avg_ratings_all = {1:(2.500, 3.947), 2:(1.714, 3.850), 3:(2.250, 4.158), 4:(2.900, 3.765), 5:(2.111, 4.111), 
        6:(2.889, 3.944), 7:(2.500, 3.913), 8:(2.929, 4.000), 9:(2.222, 3.833), 10:(2.154, 4.000),
        11:(1.909, 4.125), 12:(2.444, 4.222), 13:(2.000, 3.933), 14:(2.833, 4.133), 15:(2.286, 4.000), 
        16:(2.636, 4.125), 17:(2.500, 4.263), 18:(2.750, 4.105), 19:(1.700, 4.059), 20:(2.714, 3.923), 
        21:(2.444, 3.833), 22:(2.000, 4.150)
    }

    DATASIZE = 10 

    user_models, regression_models = {}, {}

    def __init__(self):
        # each rating pairs are:
        # (rating when people said 'yes', rating when people said 'no')
        # where 5 means high satisfactory quality and 1 means dissatisfacton
        yns = self.d_yns
        v1 = yns[1]

        files = [ 'input/d_confidence_rating_data.in', 'input/h_confidence_rating_data.in']
        for input_file in files:
            #avg_ratings = self.d_avg_ratings if input_file[6] == "d" else self.hoh_avg_ratings
            avg_ratings = self.d_avg_ratings_all if input_file[6] == "d" else self.hoh_avg_ratings_all

            function_list = {} # where all SDT functions gets stored.
            with open(input_file) as f:
                content = f.readlines()
                fa_line = content[0]
                fa_list = list(map(lambda x: x.strip('\n'), fa_line.split("\t")))
                fa_list = fa_list[:1] + list(map(lambda x: float(x), fa_list[1:])) # conversion to float
                false_alarm, correct_rejection = sum(fa_list[1:6]), sum(fa_list[6:]) # get summary values for fa and cr
                for i in range(1, len(fa_list[1:])):
                    fa_list[i] = fa_list[i] / (false_alarm + correct_rejection)
                
                #for line in range(1, len(content)-1, 1): # from 2 to 22, because v1 would be FA
                for line in range(0, 9, 1): # from 1 to 9, because v1 would be FA and limit to single factor variation
                    hit_line = content[line]
                    hit_list = list(map(lambda x: x.strip('\n'), hit_line.split("\t")))
                    v_name = hit_list[0]
                    hit_list = hit_list[:1] + list(map(lambda x: float(x), hit_list[1:])) #convert to float
                    hit, miss = sum(hit_list[1:6]), sum(hit_list[6:]) # get hit and miss
                    
                    sdt_obj = SDT(HI=hit, MI=miss, CR=correct_rejection, FA=false_alarm)
                    
                    # Now lets create a list with the rates of hit and false alarm
                    for i in range(1, len(hit_list[1:])):
                        hit_list[i] = hit_list[i] / (hit + miss)

                    tpr, fpr = hit_list[1:], fa_list[1:]
                    tpr_cum, fpr_cum = get_cumul_z(tpr), get_cumul_z(fpr)
                    coefs = poly.polyfit(fpr, tpr, 2) # Polynomial fitting line, this can be used instead of the default line.                
                    x = np.linspace(0, 1, 10)
                    ffit = poly.polyval(x, coefs)
                    #print(v_name, coefs, ffit)

                    #To create a frozen distribution:
                    p_h, p_m, p_fa, p_cr = hit/(hit+miss), miss/(hit+miss), false_alarm/(false_alarm+correct_rejection), correct_rejection/(false_alarm+correct_rejection)
                    pv_name = int(v_name.split(":")[0][1:]) #parse_v_name(v_name)
                    

                    if sdt_obj.dprime() > 0:
                        function_list[pv_name] = sdt_obj
                        #print(pv_name, "p(H)=" + str(p_h) + "\td'=" + str(sdt_obj.dprime()), "\tc=" + str(sdt_obj.c()))                    
                    else:
                        function_list[pv_name] = sdt_obj
                        #print(v_name, "p(H)=" + str(p_h))

            
            """--- file closed at this point ---"""
            reg_function_list = {} # all regression models were stored here.
            for key in function_list:
                reg_function_list = self.get_regression_model(function_list, key, reg_function_list, avg_ratings)        

            # now, let's put them into global dictionary (array)                
            self.user_models[input_file[6:7]] = function_list # per each hearing group
            self.regression_models[input_file[6:7]] = reg_function_list
        ### OUT OF THE LOOP ###
    
    def get(self):
        return (self.user_models, self.regression_models)

    # Visualizing the Polymonial Regression results
    def viz_polymonial(self, y, polynom_feat, pfa):
        X = np.linspace(0, 1, 100)
        _X = []
        for i in X:
            _X.append([i, pfa])
        _X = polynom_feat.fit_transform(_X)
        plt.plot(X, y.predict(_X), color='blue')
        plt.show()
        return


    def get_regression_model(self, function_list, key, reg_function_list, avg_ratings):
        #print(key, function_list[key])
        sdt_obj = function_list[key]
        # generate normal curves
        noi_d = stats.norm(loc=0, scale=1)
        sig_d = stats.norm(loc=sdt_obj.dprime(), scale=1) #where loc is the mean and scale is the std dev
        # estimated rates
        epm = sig_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c()) # estimated_probability_of_miss
        epcr = noi_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c()) # estimated_probability_of_correctrejection
        eph = 1-epm
        epfa = 1-epcr
        ####################################################################
        # let's find a function that can predict ratings from hit rates.
        ptset_ph, ptset_y = [0, 1], [1, 5]
        #ptset_ph, ptset_y = [], []
        # each point has a tuple (p(H), Rating).

        #print("p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} => R:{}".format(eph, epm, epfa, avg_ratings[key]))
        # now, let's add two more points from user data, but only the meaningful ones.
        if sdt_obj.dprime() > 0:
            ptset_ph.append(eph)
            ptset_y.append(avg_ratings[key][0])

            # ptset_ph.append(epm)
            # ptset_y.append(avg_ratings[key][1])
        # else:
        #     ptset_ph, ptset_y = [0, 1], [1, 5]

        ptset_pfa = [] # Let's include pfa
        for i in range(len(ptset_ph)):  # to match the number of points we have
            ptset_pfa.append(epfa)

        #  now this becomes a multivariate regression model    
        X = [] # X is the independent variable (bivariate in this case)
        for i in range(len(ptset_ph)):
            X.append([ptset_ph[i], ptset_pfa[i]])
        polynom_feat = PolynomialFeatures(degree=2)        #generate a model of polynomial features        
        X_ = polynom_feat.fit_transform(X) #transform the x data for proper fitting (for single variable type it returns, [1, x, x**2])
        
        # Preform the actual regression
        reg = linear_model.LinearRegression() # generate the regression object
        reg.fit(X_, ptset_y) # ptset_y is the dependent data
        
        #self.viz_polymonial(reg, polynom_feat, epfa)

        reg_function_list[key] = reg # add the regression model to the list.        
        return reg_function_list




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

def get_truncated_normal(mean=0, sd=0, low=0, high=10):
    value = truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
    return value

def parse_v_name(vname):
    """v_names are following 
    V2: Delay 6 sec."""
    v = int(vname.split(":")[0][1:])
    variations = {
        # main effects
        # delay, speed, number of missing words, (nothing=0, high-frequency=1, low-frequency=2), (verbatim=0, paraphrased=1)
        1: (3000, 160, 0, 0, 0),      # (3000 ms, 140-160 wpm, 0 missing words, verbatim) -> best case
        2: (6000, 160, 0, 0, 0),      # (6000 ms, 140-160 wpm, 0 missing words, verbatim)
        3: (3000, 90, 0, 0, 0),       # (3000 ms, 90 wpm, 0 missing words, verbatim)
        4: (3000, 200, 0, 0, 0),      # (3000 ms, 200 wpm, 0 missing words, verbatim)
        5: (3000, 160, 1, 1, 0), # (3000 ms, 140-160 wpm, 1 high-freq missing words, verbatim) 
        6: (3000, 160, 5, 1, 0), # (3000 ms, 140-160 wpm, 5 high-freq missing words, verbatim)
        7: (3000, 160, 1, 2, 0), # (3000 ms, 140-160 wpm, 1 low-freq missing words, verbatim)
        8: (3000, 160, 5, 2, 0), # (3000 ms, 140-160 wpm, 5 low-freq missing words, verbatim)
        9: (3000, 160, 0, 0, 1),      # (3000 ms, 140-160 wpm, 0 missing words, paraphrased)
    }
    print(v, variations[v])
    return v
    



    # delay, speed, number of missing words, (nothing=0, high-frequency=1, low-frequency=2), (verbatim=0, paraphrased=1)
    # let's generate a random input
    