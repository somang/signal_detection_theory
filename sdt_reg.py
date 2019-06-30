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

yns = { 
    1:(20, 32), 2:(18, 34), 3:(17, 35), 4:(19, 33), 5:(17, 35), 6:(22, 30),
    7:(16, 36), 8:(23, 29), 9:(22, 30), 10:(25, 27), 11:(20, 32), 12:(18, 34),
    13:(23, 29), 14:(27, 25), 15:(29, 23), 16:(23, 29), 17:(16, 36), 18:(17, 35),
    19:(22, 30), 20:(28, 24), 21:(20, 32), 22:(19, 33)
}

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


v1 = yns[1]
for v in range(2, 5): # 23):
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

    print("v{}: {}".format(v, sdt_obj))
    #print("p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} p(CR):{:.3f}".format(ph, pm, pfa, pcr))
    #print("d'={:.3f},   d'set={}".format(sdt_obj.dprime(), ds[v]))
    #print("c={:.3f},   c set={}".format(sdt_obj.c(), cs[v]))
    #print("d'={}, c={}".format(sdt_obj.dprime(), sdt_obj.c()))

    print("p(H):{:.3f} p(M):{:.3f} p(FA):{:.3f} p(CR):{:.3f}".format(eph, epm, epfa, epcr))

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


