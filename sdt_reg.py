import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

import numpy as np
import numpy.polynomial.polynomial as poly

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from cycler import cycler


from scipy import stats
from scipy.stats import norm

import math
from math import exp,sqrt,pi

from line import Line
import re

#To create a frozen distribution:
import scipy.stats
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1)

ds = {
    2:-0.102, 3:-0.155, 4:-0.051, 5:-0.155, 6:0.099, 7:-0.209, 8:0.148, 9:0.099, 10:0.245,
    11:0.000, 12:-0.102, 13:0.148, 14:0.342, 15:0.439, 16:0.148, 17:-0.209, 18:-0.155, 19:0.099,
    20:0.390, 21:0.000, 22:-0.051
}

cs = {
    2:0.345, 3:0.371, 4:0.319, 5:0.371, 6:0.244, 7:0.398, 8:0.219, 9:0.244, 10:0.171,
    11:0.293, 12:0.345, 13:0.219, 14:0.123, 15:0.074, 16:0.219, 17:0.398, 18:0.371, 19:0.244,
    20:0.098, 21:0.293, 22:0.319
}

yns = { 
    1:(20, 32), 2:(18, 34), 3:(17, 35), 4:(19, 33), 5:(17, 35), 6:(22, 30),
    7:(16, 36), 8:(23, 29), 9:(22, 30), 10:(25, 27), 11:(20, 32), 12:(18, 34),
    13:(23, 29), 14:(27, 25), 15:(29, 23), 16:(23, 29), 17:(16, 36), 18:(17, 35),
    19:(22, 30), 20:(28, 24), 21:(20, 32), 22:(19, 33)
}

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

d_avg_ratings = {
    6: (2.000, 4.083), 9: (1.769, 3.833), 14: (2.133, 3.800), 15: (1.800, 3.900), 20: (1.929, 3.636)
}

# each rating pairs are:
#                               (rating when people said 'yes', rating when people said 'no')
# where 5 means high satisfactory quality and 1 means dissatisfacton
hoh_avg_ratings = {
    4: (2.900, 3.765), 5: (2.111, 4.111), 6: (2.889, 3.944), 8: (2.929, 4.000), 9: (2.222, 3.833), 10: (2.154, 4.000),
    11: (1.909, 4.125), 12: (2.444, 4.222), 13: (2.000, 3.933), 14: (2.833, 4.133), 15: (2.286, 4.000), 16: (2.636, 4.125), 
    19: (1.700, 4.059), 20: (2.714, 3.923), 21: (2.444, 3.833)
}

v1 = yns[1]
for v in range(2, 23):
    h, m, fa, cr = yns[v][0], yns[v][1], v1[0], v1[1]
    sdt_obj = SDT(HI=h, MI=m, FA=fa, CR=cr)
    ph, pm, pfa, pcr = h/(h+m), m/(h+m), fa/(fa+cr), cr/(fa+cr)
    
    # generate
    noi_d = scipy.stats.norm(loc=0, scale=1)
    sig_d = scipy.stats.norm(loc=sdt_obj.dprime(), scale=1) #where loc is the mean and scale is the std dev
    # estimated rates
    epm = sig_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c())
    epcr = noi_d.cdf(sdt_obj.dprime()/2 + sdt_obj.c())
    eph = 1-epm # sig_d.sf(sdt_obj.c())
    epfa = 1-epcr # noi_d.sf(sdt_obj.c())
    # to the right of the criterion of signal should be hits -> using sf because it's greater
    # to the left of the criterion of signal should be misses -> using cdf because it's less than c
    # to the right of the criterion of noise should be false_alarm -> using sf because it's greater
    # to the left of the criterion of noise should be correct_rejection -> using cdf because it's less than c

    #print("v{}: {}".format(v, sdt_obj))
    #print("p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} p(CR):{:.3f}".format(eph, epm, epfa, epcr))

    #print("diff in p(h) = {:.2f} % and p(fa) = {:.2f} % \n".format( (ph-eph)*100, (pfa-epfa)*100 ))

    # To find the probability that the variable has a value LESS than or equal
    # you'd use CDF cumulative Density Function
    # a = sig_d.cdf(sdt_obj.c())
    # print("{:.3f}, {:.3f}".format(a, 1-a))
    # To find the probability that the variable has a value GREATER than or
    # equal to, you'd use SF Survival Function
    # c = sig_d.sf(sdt_obj.c())
    # print("{:.3f}, {:.3f}".format(c, 1-c))
    
    # x = np.linspace(noi_d.ppf(0.01), noi_d.ppf(0.99), 100)
    # ax.plot(x, noi_d.pdf(x))
    # y = np.linspace(sig_d.ppf(0.01), sig_d.ppf(0.99), 100)
    # ax.plot(y, sig_d.pdf(y), ls=':', c='r')
    # ax.axvline(x=sdt_obj.c(), ls='--')
    # plt.show()


    """ There should be a way to develop a prediction function, which
    takes the p(H), p(M), p(FA) to predict ratings, R.
    We know that
                        high p(H) implies low R and
                        high p(M) implies high R
    
    The fitting line, then should include
                        dprime, c -> p(H) and p(FA) -> rating R
    """

    ####################################################################
    # let's find a function that can predict ratings from hit rates.
    #######################
    # deaf group first.
    if v in d_avg_ratings.keys():
        ptset_x = [0, 1]
        ptset_y = [5, 0]
        # each point has a tuple (p(H), Rating).

        print("{}: p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} => R:{}".format(v, eph, epm, epfa, d_avg_ratings[v]))
        # now, let's add two more points.
        ptset_x.append(eph)
        ptset_y.append(d_avg_ratings[v][0])
        ptset_x.append(epm)
        ptset_y.append(d_avg_ratings[v][1])

        # then, now we have four points ready to be fitted...
        # what about p(FA) rates then?
        coefs = poly.polyfit(ptset_x, ptset_y, 2) # Fit with polyfit
        print(coefs)
        f = np.poly1d(coefs)
        print(f)
        x_new = np.linspace(0, 1, 100)
        predicted_ratings = poly.polyval(x_new, coefs)
        plt.plot(x_new, predicted_ratings)
        plt.scatter(ptset_x, ptset_y, marker='.', color="red")
        plt.ylabel('predicted_ratings')
        plt.xlabel('hit_rates')
        plt.show()



