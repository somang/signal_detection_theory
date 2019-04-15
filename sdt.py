import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

import numpy as np
import numpy.polynomial.polynomial as poly

from sklearn import metrics
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from scipy.stats import norm
from math import exp,sqrt



Z = norm.ppf
#fig, ax = plt.subplots(1, 1)


#The probability density function for norm is:
# norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi) for a real number x.
# The probability density above is defined in the “standardized” form. 
# To shift and/or scale the distribution use the loc and scale parameters. 
# Specifically, norm.pdf(x, loc, scale) is identically equivalent to
# norm.pdf(y) / scale with y = (x - loc) / scale.


# if you accidentally write scipy.stats.norm(mean=100, std=12) instead of
#  scipy.stats.norm(100, 12) or scipy.stats.norm(loc=100, scale=12), then it'll accept it,
#  but silently discard those extra keyword arguments and give you the default (0,1)

# print(norm.cdf(x, mean, sd))

"""
Here is more info. 
First you are dealing with a frozen distribution
 (frozen in this case means its parameters are set to specific values). 
 
 To create a frozen distribution:

import scipy.stats
scipy.stats.norm(loc=100, scale=12)
#where loc is the mean and scale is the std dev
#if you wish to pull out a random number from your distribution
scipy.stats.norm.rvs(loc=100, scale=12)

#To find the probability that the variable has a value LESS than or equal
#let's say 113, you'd use CDF cumulative Density Function
scipy.stats.norm.cdf(113,100,12)
Output: 0.86066975255037792
#or 86.07% probability

#To find the probability that the variable has a value GREATER than or
#equal to let's say 125, you'd use SF Survival Function 
scipy.stats.norm.sf(125,100,12)
Output: 0.018610425189886332
# or 1.86%

#To find the variate for which the probability is given, let's say the 
#value which needed to provide a 98% probability, you'd use the 
#PPF Percent Point Function
scipy.stats.norm.ppf(.98,100,12)
Output: 124.64498692758187


print(norm.sf(0, 0, 1))
print(norm.sf(0, 1.075, 1))


# Display the probability density function (pdf)
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')

# the distribution object can be called (as a function) to fix the shape, location and scale parameters. 
# This returns a “frozen” RV object holding the given parameters fixed.
#Freeze the distribution and display the frozen pdf:
rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# Check accuracy of cdf and ppf:
vals = norm.ppf([0.001, 0.5, 0.999])
print(np.allclose([0.001, 0.5, 0.999], norm.cdf(vals)))

# Generate random numbers:
r = norm.rvs(size=1000)
# And compare the histogram:
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()


matrix convolution

"""


"""
https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0
"""
def pdf(x, sd, mean):
    const = 1.0 / np.sqrt(2*np.pi*(sd**2))
    pdf_normal_dist = const*np.exp(-((x-mean)**2)/(2.0*(sd**2)))
    return pdf_normal_dist

def plot_pdf(ideal_pdf, error_pdf, ax):
    ax.fill(x, ideal_pdf, "g", alpha=0.5)
    ax.fill(x, error_pdf,"r", alpha=0.5)
    ax.set_xlim([0,1])
    ax.set_ylim([0,5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_xlabel('Z-score', fontsize=12)
    ax.legend(["noise (ideal)", "error"])

def plot_roc(good_pdf, bad_pdf, ax):
    #Total
    total_bad = np.sum(bad_pdf)
    total_good = np.sum(good_pdf)
    #Cumulative sum
    cum_TP = 0
    cum_FP = 0
    #TPR and FPR list initialization
    TPR_list=[]
    FPR_list=[]
    #Iteratre through all values of x
    for i in range(len(x)):
        #We are only interested in non-zero values of bad
        if bad_pdf[i]>0:
            cum_TP+=bad_pdf[len(x)-1-i]
            cum_FP+=good_pdf[len(x)-1-i]
        FPR=cum_FP/total_good
        TPR=cum_TP/total_bad
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    #Calculating AUC, taking the 100 timesteps into account
    auc=np.sum(TPR_list)/100
    #Plotting final ROC curve
    ax.plot(FPR_list, TPR_list)
    ax.plot(x,x, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f"%auc])


def dPrime(hits, misses, cr, fa):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/(hits+misses)
    halfFa = 0.5/(fa+cr)
 
    # Calculate hitrate and avoid d' infinity
    hitRate = hits/(hits+misses)
    if hitRate == 1: hitRate = 1-halfHit
    if hitRate == 0: hitRate = halfHit
 
    # Calculate false alarm rate and avoid d' infinity
    faRate = fa/(fa+cr)
    if faRate == 1: faRate = 1-halfFa
    if faRate == 0: faRate = halfFa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp((Z(faRate)**2 - Z(hitRate)**2)/2)
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out
    # Note the adjustment of rate=0 and rate=1, to prevent infinite values.

def plot_roc_curve(fpr, tpr, var, p=False): # takes false-positive rate, true-positive rate 
    if p:
        x = np.linspace(0, 1, 100)
        coefs = poly.polyfit(fpr, tpr, 2) # Fit with polyfit
        ffit = poly.polyval(x, coefs)
        plt.plot(x, ffit, color='green', label='polyfit') # polyfit - fitting line with the dots.
        #print('number of points:', len(fpr))        
        #print(np.poly1d(coefs[::-1]))
    
    plt.plot(fpr, tpr, color='orange', label='ROC', marker='.') #, linestyle="None")
    # guide line     
    plt.plot([0, 1], [0, 1], color='lightblue', linestyle='--', label="d'=0")
    plt.xlabel('False-Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title(var + "-" + 'Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    #plt.show()

    file_name = var.split("_")[0] + "/" + var + '.png'
    print(file_name)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    """
    d_prime = dPrime(18, 12, 3, 27)
    print(d_prime) # should be -153.49%

    ### using sdt_metrics library
    hi,mi,cr,fa = 18, 12, 3, 27
    print(dprime(hi,mi,cr,fa))

    sdt_obj = SDT(HI=18, MI=12, CR=3, FA=27)
    #print(sdt_obj)
    print(sdt_obj.c())
    print(sdt_obj.dprime())
    """

    """
    y_actual = np.array([1, 1, 2, 2])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_scores, pos_label=2)
    roc_auc = roc_auc_score(y_actual, y_scores)
    print(fpr)
    print(tpr)
    print(thresholds)
    print(roc_auc)
    """

    ########################################################################################################################
    files = ['a_rating_data.in', 'd_rating_data.in', 'h_rating_data.in']
    for input_file in files:
        print("processing.....................................", input_file)
        with open(input_file) as f:
            content = f.readlines()
            for i in range(0, len(content), 2):
                hit_line, fa_line = content[i+1], content[i]
                hit_list = list(map(lambda x: x.strip('\n'), hit_line.split("\t"))) # map(function_to_apply, list_of_inputs)            
                fa_list = list(map(lambda x: x.strip('\n'), fa_line.split("\t")))          
                print(hit_list[0], fa_list[0])
                tpr, fpr = hit_list[1:], fa_list[1:]
                tpr = list(map(lambda x: float(x), tpr))
                fpr = list(map(lambda x: float(x), fpr))
                fname = input_file.split('_')[0] + "_" + hit_list[0].replace(":", "_") # a_V1_No Delay error (3 sec.)        
                plot_roc_curve(fpr, tpr, fname, True)
    
    
                
    
    """
    
    tpr = [0.299, 0.598, 0.790, 0.931, 0.990, 1] # hit rate: y-axis of ROC
    fpr = [0.019, 0.120, 0.253, 0.565, 0.834, 1] # false-alarm rate: x-axis of ROC
    plot_roc_curve(fpr, tpr, 'testing.', True)  
    

    
    ######################################################
    x = np.linspace(0, 1, num=100)
    good_pdf = pdf(x, 0.1, 0.4)
    bad_pdf = pdf(x, 0.1, 0.6)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    plot_pdf(good_pdf, bad_pdf, ax[0])
    plot_roc(good_pdf, bad_pdf, ax[1])
    plt.tight_layout()
    plt.show()

    ######################################################
    x = np.linspace(0, 1, num=100)
    fig, ax = plt.subplots(3,2, figsize=(10,12))
    means_tuples = [(0.5,0.5),(0.4,0.6),(0.3,0.7)]
    i=0
    for good_mean, bad_mean in means_tuples:
        good_pdf = pdf(x, 0.1, good_mean)
        bad_pdf  = pdf(x, 0.1, bad_mean)
        plot_pdf(good_pdf, bad_pdf, ax[i,0])
        plot_roc(good_pdf, bad_pdf, ax[i,1])
        i+=1
    plt.tight_layout()
    """