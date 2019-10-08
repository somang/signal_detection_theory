import csv
import numpy as np
import numpy.polynomial.polynomial as poly

import matplotlib.pyplot as plt
import random
from random import randint
from random import gauss
from random import choices

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
#print("SCALE:",SCALE,", SIZE:",DATASIZE)

def test_rating_functions():
    print("TESTING.... WITH PREDEFINED VALUES...")
    test_delay = [0, 3000, 6000, 9000, 12000] # 0, 3, 6, 9, 12 sec...
    for delay in test_delay:
        delay_rating = get_delay_rating(hearing_group, v2_rm, v2_um, delay_map_function.solve(delay), delay)
    
    test_speed = [0, 45, 90, 120, 160, 190, 200, 230, 260]
    for speed in test_speed:
        speed_rating = get_speed_rating(hearing_group, v4_rm, v4_um, speed_map_function.solve(speed), speed)
    
    test_missingwords = [0, 1, 2, 4, 5, 10, 20, 50]
    for missingword_count in test_missingwords:
        get_mw_rating(hearing_group, v6_rm, v6_um, mw_map_function.solve(missingword_count), missingword_count)
    
    test_paraphrasing = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    for pf_value in test_paraphrasing:
        get_paraphrasing_rating(hearing_group, v9_rm, v9_um, pf_map_function.solve(pf_value), pf_value)

#https://crtc.gc.ca/eng/archive/2012/2012-362.htm
def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  value = truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
  return value

def data_gen():
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
            mw = np.rint(trn.rvs()) # round to integer value.
            missing_words.append(mw)
        else:               # Verbatim -> no missing words.
            missing_words.append(0)
    r_paraphrasing = np.asarray(paraphrasing)
    r_missing_words = np.asarray(missing_words)
    r_paraphrasing = np.asarray(paraphrasing)
    
    c = np.column_stack((r_delay, r_wpm)) #first two columns, then
    c = np.column_stack((c, r_missing_words))
    c = np.column_stack((c, r_paraphrasing))
    np.random.shuffle(c) # shuffle the order in rows
    ###### Simulated scores based on the fact generated from previous. ######
    #mw_trn = get_truncated_normal(mean=4.26, sd=2.32, low=0.0, high=10)
    return c

def get_probabilities(dprime, c):
    # generate normal curves
    noi_d = stats.norm(loc=0, scale=1)
    sig_d = stats.norm(loc=dprime, scale=1) # where loc is the mean and scale is the std dev
    # estimated rates
    epm = sig_d.cdf(dprime/2 + c)           # estimated p(m)
    eph = 1-epm # similar to sig_d.sf(c)    # estimated p(h)
    epcr = noi_d.cdf(dprime/2 + c)          # estimated p(cr)
    epfa = 1-epcr # similar to noi_d.sf(c)  # estimated p(fa)
    eph = 0 if eph < 0 else 1 if eph > 1 else eph
    return (eph, epfa)

def testprinting(factorvalue, zscore_val, eph, rating):
    print("raw:{}, Z-score (c):{}, \np(H):{} ==> Predicted Rating:{}\n".format(factorvalue, zscore_val, eph, rating))

def get_delay_rating(group, reg_function, user_model, zscore_val, delay):    
    rating = 0
    regression_mode = 1 if user_model.dprime() > 1 else 0
    eph, epfa = get_probabilities(user_model.dprime(), zscore_val)
    if regression_mode:
        X_test = PolynomialFeatures(degree=2).fit_transform([ [eph, epfa] ]) 
        if delay > 6000:
            rating = 1
        elif delay < 100: # minimum bound?
            rating = 5
        else:
            rating = reg_function.predict(X_test)    
    else:                                   # manual fitting, polynomial on raw values...                
        if group == 'd':
            v1_rating = choices([2.250, 3.846], [0.480, 0.520])[0] # values, probabilities of that choice
            v2_rating = choices([1.818, 4.071], [0.440, 0.560])[0]
        else:                               # group == 'h'
            v1_rating = choices([2.500, 3.947], [0.296, 0.704])[0]
            v2_rating = choices([1.714, 3.850], [0.259, 0.741])[0]        
        polyfit_x, polyfit_y = [0, 3000, 6000, 9000], [5, v1_rating, v2_rating, 1]        
        rating_reg = poly.polyfit(polyfit_x, polyfit_y, 2)
        rating = poly.polyval(delay, rating_reg)
    rating = 1 if rating < 1 else 5 if rating > 5 else rating
    #testprinting(delay, zscore_val, eph, rating)
    return rating
    
def get_speed_rating(group, reg_function, user_model, zscore_val, speed):
    rating = 0 # let's fix axis for v4 only. next would be to split into slow and fast signals
    #regression_mode = 1 if user_model.dprime() > 1 else 0
    regression_mode = 0 # because it doesn't make sense with v4 only to define the rating algorithm.

    eph, epfa = get_probabilities(user_model.dprime(), zscore_val)
    if regression_mode:
        X_test = PolynomialFeatures(degree=2).fit_transform([ [eph, epfa] ])        
        if speed > 220 or speed < 45:
            rating = 1
        # elif speed < 100: # minimum bound?
        #     rating = 5
        else:
            rating = reg_function.predict(X_test)
    else:                           # manual fitting, polynomial on raw values.
        if group == 'd':
            v3_rating = choices([1.889, 3.688], [0.360, 0.640])[0] # V3: 90 WPM
            v4_rating = choices([2.333, 4.000], [0.360, 0.640])[0] # V4: 200 WPM
        else:                       # group == 'h'
            v3_rating = choices([2.250, 4.158], [0.296, 0.704])[0]
            v4_rating = choices([2.900, 3.765], [0.370, 0.630])[0]        
        polyfit_x, polyfit_y = [0, 90, 160, 200, 250], [1, v3_rating, 5, v4_rating, 1]
        rating_reg = poly.polyfit(polyfit_x, polyfit_y, 2)
        rating = poly.polyval(speed, rating_reg)
    rating = 1 if rating < 1 else 5 if rating > 5 else rating
    #testprinting(speed, zscore_val, eph, rating)
    return rating

def get_mw_rating(group, reg_function, user_model, zscore_val, mw_count):    
    rating = 0
    #regression_mode = 1 if user_model.dprime() > 1 else 0
    regression_mode = 0 # because it doesn't make sense with v4 only to define the rating algorithm.

    eph, epfa = get_probabilities(user_model.dprime(), zscore_val)
    if regression_mode:
        X_test = PolynomialFeatures(degree=2).fit_transform([ [eph, epfa] ])        
        if mw_count > 10:
            rating = 1
        elif mw_count == 0:
            rating = 5
        else:
            rating = reg_function.predict(X_test)
    else: # manual fitting, polynomial on raw values...
        if group == 'd':        # choices(values, probabilities_of_the_values_occur) of that choice
            v5_rating = choices([1.750, 3.529], [0.320, 0.680])[0]
            v6_rating = choices([2.000, 4.083], [0.520, 0.480])[0]
            v7_rating = choices([2.333, 3.923], [0.480, 0.520])[0]
            v8_rating = choices([2.333, 4.062],	[0.360, 0.640])[0]
        else:                   # group == 'h'
            v5_rating = choices([2.111, 4.111], [0.333, 0.667])[0]
            v6_rating = choices([2.889, 3.944], [0.333, 0.667])[0]
            v7_rating = choices([2.500, 3.913], [0.148, 0.852])[0]
            v8_rating = choices([2.929, 4.000],	[0.519, 0.481])[0]
        polyfit_x, polyfit_y = [0, 1, 5, 10], [5, v5_rating, v6_rating, 1] # because v6:5HF has positive sensitivity in both groups...
        rating_reg = poly.polyfit(polyfit_x, polyfit_y, 2)
        rating = poly.polyval(mw_count, rating_reg)
    rating = 1 if rating < 1 else 5 if rating > 5 else rating
    #testprinting(mw_count, zscore_val, eph, rating)
    return rating

def get_paraphrasing_rating(group, reg_function, user_model, zscore_val, pf_value):
    rating = 0
    eph, epfa = get_probabilities(user_model.dprime(), zscore_val)
    if group == 'd':            # choices(values, probabilities_of_the_values_occur) of that choice
        v9_rating = choices([1.769, 3.833], [0.520, 0.480])[0]
    else:                       # group == 'h'
        v9_rating = choices([2.222, 3.833], [0.333, 0.667])[0]        
    polyfit_x, polyfit_y = [0, 1], [5, v9_rating] # because v6:5HF has positive sensitivity in both groups...
    rating_reg = poly.polyfit(polyfit_x, polyfit_y, 2)
    rating = poly.polyval(pf_value, rating_reg)
    rating = 1 if rating < 1 else 5 if rating > 5 else rating
    return rating
 
if __name__ == "__main__":
    test = 0 # run testing lines...
    
    data_cols = data_gen() # Generate random values.
    # Use the user model to generate quality ratings
    # The definition of 'rating system' will be incorperating the user models.
    user_model, reg_model = Usermodels().get()    # Variable user_models will have a "SDT object" and a "quality regression model"
    
    hearing_group = 'd'
    #hearing_group = 'h'
    v1 = user_model[hearing_group][1]
    pfa = v1['FA']/int(v1['FA']+v1['CR'])
    
    # 1. Delay tools
    v2_um, v2_rm = user_model[hearing_group][2], reg_model[hearing_group][2]
    if v2_um.dprime() > 0:
        delay_map_function = Line( (3000, v2_um.c()-v2_um.dprime()) , (6000, v2_um.c()) ) # linear mapping...
    else:
        delay_map_function = Line( (0, 0) , (6000, v2_um.c()) ) # linear mapping...
    # 2. Speed tools
    v3_um, v3_rm = user_model[hearing_group][3], reg_model[hearing_group][3] # slow
    v4_um, v4_rm = user_model[hearing_group][4], reg_model[hearing_group][4] # fast
    # regression model for v4:200 wpm as for deaf , the d' is the same for v3 and v4.
    if v4_um.dprime() > 0:
        speed_map_function = Line( (160, v4_um.c()-v4_um.dprime()) , (200, v4_um.c()) ) # linear mapping...
    else:
        speed_map_function = Line( (0, 0) , (200, v4_um.c()) ) # linear mapping...
    # 3. Missing Word count
    mapping = 1

    v5_um, v5_rm = user_model[hearing_group][5], reg_model[hearing_group][5] # 1HF d:-, h:+
    v6_um, v6_rm = user_model[hearing_group][6], reg_model[hearing_group][6] # 5HF d:+, h:+  -> this can be used..
    v7_um, v7_rm = user_model[hearing_group][7], reg_model[hearing_group][7] # 1LF d:-, h:-
    v8_um, v8_rm = user_model[hearing_group][8], reg_model[hearing_group][8] # 5LF d:-, h:+
    if mapping == 2: # v6_um.dprime() > 0:
        mw_map_function = Line( (0, v6_um.c()-v6_um.dprime()) , (5, v4_um.c()) ) # linear mapping... using dprime
    else:
        mw_map_function = Line( (0, 0) , (5, v4_um.c()) ) # linear mapping...
    # 4. Paraphrasing
    v9_um, v9_rm = user_model[hearing_group][9], reg_model[hearing_group][9] # paraphrasing
    pf_map_function = Line( (0, 0) , (1, v9_um.c()) ) # linear mapping...
    
    #test_rating_functions() # turn on the printing function if to test these...

    # generate rating scores based on the random input.
    for c in data_cols:
        delay, speed, missingword_count, pf_value = c[0], c[1], c[2], c[3]
        ######### GET RATINGS ####################################################################
        # 1. Delay
        # Delay did not have a positive sensitivity for both hearing groups, however, we may start from p(H) of both delays.
        # Let's find p(H) for 3 sec (variation 1) and 6 sec delay (variation 2)    
        delay_rating = get_delay_rating(hearing_group, v2_rm, v2_um, delay_map_function.solve(delay), delay)
        # 2. Speed
        # Speed for deaf group, didn't have positive sensitivity
        # however, hoh group had the positive sensitivity on V4 (fast speed)
        speed_rating = get_speed_rating(hearing_group, v4_rm, v4_um, speed_map_function.solve(speed), speed)
        # 3. Missing Words
        # Linear for now... and v6 was selected because it has the positive sensitivity from both group.
        # Word frequency would play a role in the actual predictions though...
        mw_rating = get_mw_rating(hearing_group, v6_rm, v6_um, mw_map_function.solve(missingword_count), missingword_count)
        # 4. Paraphrasing
        # The paraphrasing factor was defined as 1 or 0
        # Then, the value will be either 1 or 0, 
        pf_rating = get_paraphrasing_rating(hearing_group, v9_rm, v9_um, pf_map_function.solve(pf_value), pf_value)

        # print("delay:\t{}\t-> rating: {}\n".format(delay, delay_rating) + 
        # "speed:\t{}\t-> rating: {}\n".format(speed, speed_rating) + 
        #  "# of missing words:\t{}\t-> rating: {}\n".format(missingword_count, mw_rating) + 
        #  "(0=verbatim, 1=edited):\t{}\t-> rating: {}\n\n".format(pf_value, pf_rating))

        # now that we have the ratings, let's group the ratings to the columns.
        # rating_list = [delay_rating, speed_rating, mw_rating, pf_rating]
        # for r in rating_list:
            


        # p = np.asarray(RATING_LIST)
        # for i in p:
        #     c = np.column_stack((c, i))

    # print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)
    # filename = str(SCALE) + '_nd_dt_' + str(DATASIZE) + '.csv'
    # with open(filename, 'w') as mf:
    # wr = csv.writer(mf)
    # for i in c:
    #     wr.writerow(i)
