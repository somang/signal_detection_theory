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





def draw_sdt(tpr, fpr, ax): # false-positive rate (noise), true-positive rate (signal hit), in z scores
    # Display the probability density function (pdf)
    x = np.linspace(-5, 5, 100)     # define a big enough x interval 

    # get p(H) and p(FA)
    pHit, pFA = 0, 0
    for i in range(5):
        pHit += tpr[i]        

    for i in range(5):
        pFA += fpr[i]        

    # calculate d'
    zHit, zFA = norm.ppf(pHit), norm.ppf(pFA)
    dprime = zHit - zFA

    # calculate c
    c = -1 * (zHit + zFA) / 2
    
    noise_pdf = norm.pdf(x)
    error_pdf = norm.pdf(x, dprime, 1)              # get the norm.pdf for x interval
    ax.plot(x, noise_pdf, "g", alpha=0.9, label="noise")
    ax.plot(x, error_pdf, "r", alpha=0.9, label="error")

    plt.axvline(x=c, linestyle='-.') #, label="c= "+str(c))
    plt.text(c+0.01, 0.01, "c= "+str(c))

    plt.vlines(0, 0, 0.4, linestyle=':')
    plt.vlines(dprime, 0, 0.4, linestyle=':')
    plt.hlines(0.4, 0, dprime) #, label="d'= "+str(dprime))
    plt.text(dprime, 0.405, "d'= "+str(dprime), fontsize=10)


    # calculate c for each confidence level
    tpr_cum, fpr_cum = get_cumul_z(tpr), get_cumul_z(fpr)
    ztpr_r = list(map(lambda x: norm.ppf(x), tpr_cum))
    zfpr_r = list(map(lambda x: norm.ppf(x), fpr_cum))
    dprime_r = []
    c_r = []
    for i in range(len(ztpr_r)):
        if ztpr_r[i]:
            dprime_r.append(ztpr_r[i]-zfpr_r[i])
            c_r.append(-1 * ( ztpr_r[i] + zfpr_r[i] ) / 2)
    
    for i in c_r:
        if not math.isnan(i) and not math.isinf(i):
            plt.vlines(i, 0, 0.4, linestyle=':')

    # if dprime > 0:
    #     ax.fill_between(x, noise_pdf, error_pdf, where=x > c, facecolor='green', alpha=0.15, label="hit") # , interpolate=True)
    #     ax.fill_between(x, noise_pdf, 0, where=x > c, facecolor='red', alpha=0.15, label="false alarm") #, interpolate=True)
    #     ax.fill_between(x, noise_pdf, error_pdf, where=x < c, facecolor='blue', alpha=0.15, label="correct rejection") # , interpolate=True)
    #     ax.fill_between(x, error_pdf, 0, where=x < c, facecolor='orange', alpha=0.15, label="miss") #, interpolate=True)
    # else:
    #     ax.fill_between(x, error_pdf, 0, where=x > c, facecolor='green', alpha=0.15, label="hit")
    #     ax.fill_between(x, noise_pdf, error_pdf, where=x > c, facecolor='red', alpha=0.15, label="false alarm")
    #     ax.fill_between(x, error_pdf, 0, where=x < c, facecolor='blue', alpha=0.15, label="correct rejection")
    #     ax.fill_between(x, error_pdf, noise_pdf, where=x < c, facecolor='orange', alpha=0.15, label="miss", hatch="/")


    ax.set_xlim([-5,5])
    ax.set_ylim([0, 0.5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_xlabel('Z-score', fontsize=12)
    ax.legend()

    return ax

    
def plot_roc_curve(fpr, tpr, var, ax, p=False): # takes false-positive rate, true-positive rate 
    if p:
        x = np.linspace(0, 1, 100)
        coefs = poly.polyfit(fpr, tpr, 2) # Fit with polyfit
        ffit = poly.polyval(x, coefs)
        ax.plot(x, ffit, label='polyfit') # polyfit - fitting line with the dots.
    
    ax.plot(fpr, tpr, label='ROC', marker='.') #, linestyle="None")
    # guide line     
    ax.plot([0, 1], [0, 1], linestyle='--', label="d'=0")
    ax.set_xlabel('False-Alarm Rate')
    ax.set_ylabel('Hit Rate')
    ax.set_title(var + "-" + 'Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='best')
    
    return ax

def get_cumul_z(rate):
    tmp, sum = [], 0
    for i in range(len(rate)):
        sum += float(rate[i])
        tmp.append(sum)
    return tmp

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
    
    ########################################################################################################################
    files = ['a_rating_data.in'] #, 'd_rating_data.in', 'h_rating_data.in']
    for input_file in files:
        print("processing.....................................", input_file)
        with open(input_file) as f:
            content = f.readlines()
            fa_line = content[0]
            fa_list = list(map(lambda x: x.strip('\n'), fa_line.split("\t")))
            
            for i in range(1, 3): # len(content), 1):
                hit_line = content[i]
                hit_list = list(map(lambda x: x.strip('\n'), hit_line.split("\t"))) # map(function_to_apply, list_of_inputs)
                
                print(hit_list[0], fa_list[0])
                tpr, fpr = hit_list[1:], fa_list[1:]
                tpr = list(map(lambda x: float(x), tpr))
                fpr = list(map(lambda x: float(x), fpr))
                tpr_cum, fpr_cum = get_cumul_z(tpr), get_cumul_z(fpr)
                fname = input_file.split('_')[0] + "_" + hit_list[0].replace(":", "_") # a_V1_No Delay error (3 sec.)        
                
                # set grid and color 
                fig = plt.figure(tight_layout=True)
                gs = gridspec.GridSpec(3, 9)
                n = 100
                # get colormap
                cmap=plt.cm.Pastel1
                # build cycler with 5 equally spaced colors from that colormap
                c = cycler('color', cmap(np.linspace(0,1,5)) )
                # supply cycler to the rcParam
                plt.rcParams["axes.prop_cycle"] = c
                                

                # # draw roc curve
                ax = fig.add_subplot(gs[1, :3])
                plot_roc_curve(fpr_cum, tpr_cum, fname, ax, True)
                
                # draw sdt distribution
                ax = fig.add_subplot(gs[0:, 3:])
                draw_sdt(tpr, fpr, ax)
                

                plt.show()
                #file_name = var.split("_")[0] + "/" + var + '.png'
                #print(file_name)
                #f.savefig(file_name, bbox_inches='tight')
                


    # What do I need?
    # input: factor variable value x
    # output: probability of hit (H) and false-alarm (FA) given x

    # what to do?
    # convert x (ms, wpm, mw) to a z-score for the given 'signal' type.
    # then get the c value probabilities p(H) and p(FA)

    # v15_overall
    dprime = 0.330
    c = 0.111
    # v2_overall
    #dprime = -0.299
    #c = 0.426
    
    """


    
    # given two points, get the mapping equation from raw value to z-score.
    # V2: 6 sec delay
    # data = ((6, -0.299), (3, 0))
    # line = Line(data) 
    # for i in range(10):
    #     print(i, line.solve(i))

    # V3: 90 wpm speed
    # data = ((90, -0.297), (160, 0))
    # line = Line(data) 
    # for i in range(0,220,50):
    #     print(i, line.solve(i))

    # V4: 200 wpm speed
    # data = ((200, -0.175), (160, 0))
    # line = Line(data) 
    # for i in range(0,220,50):
    #     print(i, line.solve(i))

    # V5: 1 HF Missing word
    # data = ((1, -0.236), (0, 0))
    # line = Line(data) 
    # for i in range(0,10,1):
    #     print(i, line.solve(i))

    """


















    """
    
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
    

    
    To create a frozen distribution:
    import scipy.stats
    scipy.stats.norm(loc=100, scale=12)
    #where loc is the mean and scale is the sd if you wish to pull out a random number from your distribution
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
