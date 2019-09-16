import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT
import numpy as np
import numpy.polynomial.polynomial as poly
from sklearn import metrics
from sklearn.metrics import roc_auc_score

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

if __name__ == "__main__":
    ########################################################################################################################
    files = ['input/a_rating_data.in', 'input/d_rating_data.in', 'input/h_rating_data.in']
    for input_file in files:
        print("processing.....................................", input_file)
        with open(input_file) as f:
            content = f.readlines()
            fa_line = content[0]
            fa_list = list(map(lambda x: x.strip('\n'), fa_line.split("\t")))
            fa_list = fa_list[:1] + list(map(lambda x: float(x), fa_list[1:])) # conversion to float
            false_alarm, correct_rejection = sum(fa_list[1:6]), sum(fa_list[6:]) # get summary values for fa and cr
            for i in range(1, len(fa_list[1:])):
                fa_list[i] = fa_list[i] / (false_alarm + correct_rejection)
                
            for line in range(1, len(content), 1):
                hit_line = content[line]
                hit_list = list(map(lambda x: x.strip('\n'), hit_line.split("\t")))
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
                ffit = poly.polyval(x, coefs)
                print(coefs, ffit)

                # fname = input_file.split('_')[0] + "_" + hit_list[0].replace(":", "_") # a_V1_No Delay error (3 sec.)
                # fig = plt.figure(figsize=(19.2,10.8), dpi=100)
                # gs = gridspec.GridSpec(3, 9)
                # n = 100
                #ax = fig.add_subplot(gs[1, :3])
                #plot_roc_curve(fpr_cum, tpr_cum, fname, ax, True)
                #ax = fig.add_subplot(gs[0:, 3:])
                #draw_sdt(tpr, fpr, sdt_obj, ax)
                
                #if fname.split('_')[0] == 'input/h' and fname.split('_')[1] == 'V8':
                #    plt.show()
                #head = input_file.split("/")[1].split("_")[0]             
                #imgfname = re.sub(r":| ", "_", hit_list[0])
                #file_name = "img/" + head + "/" + head + "_" + imgfname + '.png'
                #print(file_name)

                #plt.savefig(file_name)
                #plt.close()


    # What do I need?
    # input: factor variable value x
    # output: probability of hit (H) and false-alarm (FA) given x

    # what to do?
    # convert x (ms, wpm, mw) to a z-score for the given 'signal' type.
    # then get the c value probabilities p(H) and p(FA)



