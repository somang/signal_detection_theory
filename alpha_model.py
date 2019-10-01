import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint
from random import gauss

from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import truncnorm
import math
from sklearn.preprocessing import PolynomialFeatures

from line import Line
from user_model import Usermodels

SCALE = 5
DATASIZE = 10
print("SCALE:",SCALE,", SIZE:",DATASIZE)

#https://crtc.gc.ca/eng/archive/2012/2012-362.htm
def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  value = truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
  return value

def get_probabilities(dprime, c):
    # generate normal curves
    noi_d = stats.norm(loc=0, scale=1)
    sig_d = stats.norm(loc=dprime, scale=1) # where loc is the mean and scale is the std dev
    # estimated rates
    epm = sig_d.cdf(dprime/2 + c)           # estimated p(m)
    epcr = noi_d.cdf(dprime/2 + c)          # estimated p(cr)
    eph = 1-epm # similar to sig_d.sf(c)    # estimated p(h)
    epfa = 1-epcr # similar to noi_d.sf(c)  # estimated p(fa)

    if c < 0: # if dprime is negative
        eph = eph * -1
        if eph < 0:
            eph = 0
        elif eph > 1:
            eph = 1

    return (eph, epfa)

if __name__ == "__main__":
    # delay, wpm, similarity, number of errors
    ### normal distribution using the mean and sd from existing data.
    trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000)
    r_delay = trn.rvs(DATASIZE)

    trn = get_truncated_normal(mean=232.03, sd=200.48, low=80, high=850) # from sample
    r_wpm = trn.rvs(DATASIZE)    
    
    # considering paraphrasing value, if paraphrased = 0, then there will be no missing words.
    # if paraphrased = 1, then we have some missing words AND the word frequency can be generated (log)    
    missing_words, paraphrasing = [], [] # zipf_scale... though
    trn = get_truncated_normal(mean=5.02, sd=6.79, low=0.0, high=25) 
    for i in range(DATASIZE):    # because it cannot have a 100% and a missing word..
        paraphrasing_value = random.randint(0, 1) # paraphrasing distribution should reflect sample analysis too. but work on this later.        
        paraphrasing.append(paraphrasing_value)
        if paraphrasing_value > 0: # Paraphrased -> some missing words.            
            # tmp_zipf_value = round(random.gauss(3, 1)) # zipf-scale is 1-7
            # if tmp_zipf_value < 1:
            #     zipf_scale_value = 1
            # elif tmp_zipf_value > 7:
            #     zipf_xcale_value = 7
            # else:
            #     zipf_scale_value = tmp_zipf_value
            # zipf_scale.append(zipf_scale_value)            

            mw = np.rint(trn.rvs()) # round to integer value.
            missing_words.append(mw)
        else:               # Verbatim -> no missing words.
            missing_words.append(0)
    r_paraphrasing = np.asarray(paraphrasing)
    r_missing_words = np.asarray(missing_words)
    r_paraphrasing = np.asarray(paraphrasing)
    
    c = np.column_stack((r_delay, r_wpm)) #first two columns, then
    c = np.column_stack((c, r_missing_words))
    #c = np.column_stack((c, r_word_freq))
    c = np.column_stack((c, r_paraphrasing))
    np.random.shuffle(c) # shuffle the order in rows
    ###### Simulated scores based on the fact generated from previous. ######
    
    #mw_trn = get_truncated_normal(mean=4.26, sd=2.32, low=0.0, high=10)

    # [delay quality rating], [speed quality rating], [missing words quality rating], [paraphrasing quality rating]
    um, rm = Usermodels().get()
    
    for i in c:
        delay_score, speed_score, verbatim_score, sge_score, missing_words_score = 0,0,0,0,0
        delay, wpm, mw, pf = i[0], i[1], i[2], i[3]
        
        #print(delay, wpm, mw, pf)
        # now that we have generated random values.
        # we should now predict quality ratings, which can be processed from p(H) and p(FA)
        
        # Variable user_models will have a "SDT object" and a "quality regression model"

    # 1. Delay
    # Delay did not have a positive sensitivity for both hearing groups, however, we may start from p(H) of both delays.
    # Let's find p(H) for 3 sec (variation 1) and 6 sec delay (variation 2)
    v1_um = um['d'][1]
    v2_um = um['d'][2]
    print(v1_um.dprime(), v1_um.c())
    print(v2_um.dprime(), v2_um.c())


    # let's try drawing the regression function first,
    regression = rm['d'][2]
    X = np.linspace(0, 1, 100)
    _X = []
    for i in X:
        _X.append([i, 0.48])
    polynom_feat = PolynomialFeatures(degree=2)
    _X = polynom_feat.fit_transform(_X)
    _Y = regression.predict(_X)
    plt.plot(X, _Y, color='blue')
    
    l = Line( ((6000, v2_um.c()) , (0, v2_um.c()-v2_um.dprime())) )

    testing_raw_delay = [0, 3000, 6000, 9000, 12000] # 0, 3, 6, 9, 12, 120 sec...
    for i in testing_raw_delay:
        eph, epfa = get_probabilities(v2_um.dprime(), l.solve(i))
        X_test = [ [eph, epfa] ]
        #print(regression.coef_)
        polynom_feat = PolynomialFeatures(degree=2)
        X_test = polynom_feat.fit_transform(X_test)
        Y_test = regression.predict(X_test)
        print("raw:{}, Z-score (c):{}, \np(H):{} ==> Predicted Rating:{}".format(i, l.solve(i), eph, Y_test))

        plt.scatter(eph, Y_test)

    plt.show()
    
    # trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000)

    # 2. Speed



    # 3. Missing Words


    # 4. Paraphrasing





    # # verbatim_score: mean=4.20, sd=2.51
    # # Paraphrasing (verbatimness) score which audiences subjectively feel
    # if sentence_sim == 1.0:
    #     verbatim_score = 10
    # elif 0.7 < sentence_sim < 1.0:
    #     if missing_words == 0:
    #     verbatim_score = 10
    #     elif 0 < missing_words <= 15:
    #     verbatim_score = round(gauss(4.50,0.3))  
    # else:
    #     verbatim_score = round(gauss(1.0,0.6))


    """
    print("====== SCORES =====")
    #print("delay score:", np.mean(c[:,6]), np.std(c[:,6]))
    #print("speed score:", np.mean(c[:,7]), np.std(c[:,7]))
    #print("sge score:", np.mean(c[:,8]), np.std(c[:,8]))
    #print("missing words scores:", np.mean(c[:,9]), np.std(c[:,9]))
    print("verbatim score:", np.mean(c[:,10]), np.std(c[:,10]))

    '''
    print("====== Actual Values =====")
    print("delay:", #min(c[:,0]), max(c[:,0]), 
        "[4075 4669.5 5775]",
        #np.mean(c[:,0]), np.std(c[:,0]),
        np.percentile(c[:,0], [25,50,75]))

    print("speed:", #min(c[:,1]), max(c[:,1]),
        "[118.56 143.21 313.46]",
        #np.mean(c[:,1]), np.std(c[:,1]),
        np.percentile(c[:,1], [25,50,75]))

    print("sge:", #min(c[:,2]), max(c[:,2]),
        #np.mean(c[:,2]), np.std(c[:,2]),
        np.percentile(c[:,2], [25,50,75]))
    
    print("missing words:", #min(c[:,3]), max(c[:,3]),
        "[0.75  1.5  7]",
        #np.mean(c[:,3]), np.std(c[:,3]),
        np.percentile(c[:,3], [25,50,75]))

    print("verbatim:", #min(c[:,4]), max(c[:,4]), 
        "[0.7770  0.8416  0.9467]",
        #np.mean(c[:,4]), np.std(c[:,4]),
        np.percentile(c[:,4], [25,50,75]))
    #print("PF factor:", min(c[:,5]), max(c[:,5]),
    #      np.mean(c[:,5]), np.std(c[:,5]),
    #      np.percentile(c[:,5], [25,50,75]))
    '''


    print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)
    filename = str(SCALE) + '_nd_dt_' + str(DATASIZE) + '.csv'
    with open(filename, 'w') as mf:
    wr = csv.writer(mf)
    for i in c:
        wr.writerow(i)
 """