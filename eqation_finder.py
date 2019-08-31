import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

import numpy as np
import numpy.polynomial.polynomial as poly

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from line import Line
import re

import csv
from xlsxwriter.workbook import Workbook

import xlrd
from xlrd import open_workbook

import pandas

import scipy.stats
from scipy import stats
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats.stats import pearsonr   
from scipy.stats.stats import kendalltau
from scipy.stats.stats import spearmanr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib import cm
from cycler import cycler

import math
from math import exp,sqrt,pi

xl = pandas.ExcelFile('q1q4_handled.xlsx')

def multiply_two(x):
    yn = x[0] # check the y/n answer
    if yn == 1: # yes errors = bad experience
        yn = -1 # make it negative.
    elif yn == -1: # no errors = good experience
        yn = 1 # make positive
    return yn*x[1]

def handle_list(df, di):
    for i, r in df.iterrows():
        if not r[0] == "uuid":
            for c in range(1,24):
                if not math.isnan(r[c]):
                    di[c].append(r[c])
    return di

def check_shapiro(yns, confs, ratings):
    for v in range(1,23):
        p = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))
        sha_y, sha_r = shapiro(p)[1], shapiro(ratings[v])[1]  
        if sha_y < 0.05:
            flag = "yns "
            if sha_r < 0.05:
                flag += "ratings"
                print("{}: {} data is not normally distributed. ({:.3f}, {:.3f})".format(v, flag, sha_y, sha_r))
        else:
            print("{} data is normal".format(v))
        
            


def print_pearson(yns, confs, ratings):
    for v in range(1,23):
        # p = yns multiply by conf then check pearson's r
        p = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))
        print(v, pearsonr(p, ratings[v])) # x, y

def print_kendall(yns, confs, ratings): # non parametric kendall's tau b
    for v in range(1,23):
        ynsconf = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))        
        t, p = kendalltau(ynsconf, ratings[v])
        if p > 0.05:
            print("{}: p > 0.05".format(v))
        else:
            if t < 0.6:
                print("{}:                  kendall's moderate t={:.3f}".format(v, t)) # x, y
            elif t >= 0.6:
                print("{}: kendall's high t={:.3f}".format(v, t))

def print_spearman(yns, confs, ratings): # non parametric
    for v in range(1,23):
        ynsconf = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))        
        r, p = spearmanr(ynsconf, ratings[v])
        if p > 0.05:
            print("{}: p > 0.05".format(v))
        else:
            if r < 0.6 and r >= 0.4:
                print("{}:                  spearman's moderate r={:.3f}".format(v, r))
            elif r >= 0.6:
                print("{}: spearman's high r={:.3f}".format(v, r))

def print_kendall_and_spearman(yns, confs, ratings):
    for v in range(1,23):
        ynsconf = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))        
        r, sp = spearmanr(ynsconf, ratings[v])
        t, kp = kendalltau(ynsconf, ratings[v])
        pr, pp = pearsonr(ynsconf, ratings[v])

        if sp > 0.05 and kp > 0.05:
            print("{}: p > 0.05, sp={}, kp={}".format(v, sp, kp))
        else:
            if (0 < r < 0.4) or (0 < t < 0.4):
                print("{}: Low - kendall t={:.3f}, spearman r={:.3f}".format(v, t, r))
            elif (0.4 <= r < 0.6) or (0.4 <= t < 0.6):
                print("{}: moderate - kendall t={:.3f}, spearman r={:.3f}".format(v, t, r))
            else:
                print("{}: high - kendall t={:.3f}, spearman r={:.3f}".format(v, t, r))


# q1 "Did you see any errors in the caption?"
# q2 "I am confident that my decision was correct:"
# q3 "How would you rate the quality of the caption?"
# q4 "Visual pleasure?"

yns, confs, ratings, vpr = {}, {}, {}, {}
d_yns, d_confs, d_ratings, d_vpr = {}, {}, {}, {}
hoh_yns, hoh_confs, hoh_ratings, hoh_vpr = {}, {}, {}, {}
for v in range(1,24):
    yns[v], confs[v], ratings[v], vpr[v] = [], [], [], []
    d_yns[v], d_confs[v], d_ratings[v], d_vpr[v] = [], [], [], []
    hoh_yns[v], hoh_confs[v], hoh_ratings[v], hoh_vpr[v] = [], [], [], []

for sn in xl.sheet_names:
    if sn == "q1_all":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)
        yns = handle_list(df, yns)
    elif sn == "q2_all":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)  
        confs = handle_list(df, confs)
    elif sn == "q3_all":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)
        ratings = handle_list(df, ratings)
    elif sn == "q4_all":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)
        vpr = handle_list(df, vpr)

    elif sn == "q1_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_yns = handle_list(df, d_yns)
    elif sn == "q2_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_confs = handle_list(df, d_confs)
    elif sn == "q3_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_ratings = handle_list(df, d_ratings)
    elif sn == "q4_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_vpr = handle_list(df, d_vpr)

    elif sn == "q1_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_yns = handle_list(df, hoh_yns)
    elif sn == "q2_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_confs = handle_list(df, hoh_confs)
    elif sn == "q3_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_ratings = handle_list(df, hoh_ratings)
    elif sn == "q4_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_vpr = handle_list(df, hoh_vpr)


# is data normal?
#check_shapiro(yns, confs, ratings)
#check_shapiro(d_yns, d_confs, d_ratings)
#check_shapiro(hoh_yns, hoh_confs, hoh_ratings)
print("OVERALL- Y/N+Confidence VS. Quality")
print_kendall_and_spearman(yns, confs, ratings)
print("DEAF- Y/N+Confidence VS. Quality")
print_kendall_and_spearman(d_yns, d_confs, d_ratings)
print("\nHOH- Y/N+Confidence VS. Quality")
print_kendall_and_spearman(hoh_yns, hoh_confs, hoh_ratings)

#print("y/n * confidence  VS visual pleasure ratings")
# is data normal?
# check_shapiro(yns, confs, vpr)
# check_shapiro(d_yns, d_confs, d_vpr)
# check_shapiro(hoh_yns, hoh_confs, hoh_vpr)

print("OVERALL- Y/N+Confidence VS. Quality")
print_kendall_and_spearman(yns, confs, vpr)
print("\n\nDEAF- Y/N+Confidence VS. VISUAL PLEASURE")
print_kendall_and_spearman(d_yns, d_confs, d_vpr)
print("\nHOH- Y/N+Confidence VS. VISUAL PLEASURE")
print_kendall_and_spearman(hoh_yns, hoh_confs, hoh_vpr)


rating = {'y':{}, 'n':{}}
for i in range(1, 23):
    rating['y'][i] = []
    rating['n'][i] = []




# for v in range(1,23):
#     put_two = list(zip(d_yns[v], d_confs[v]))
#     p = list(map(lambda x: multiply_two(x), put_two))
#     for i in range(len(p)):
#         print(p[i], d_ratings[v][i])



#         """
#         1. function g(dprime, c) -> p(H), p(FA)
#         2. given p(H) and p(FA), calcaulate the estimated quality rating
#             -> what is the relationship between 
#                 p(H): Low rating, P(M): High rating


#                 Assume hoh v8: 5 low frequency missing word
#                 d' = 0.582
#                 c = 0.244
#                 Compare the distance between
#                     high rating - low rating
#                     variation 8: (2.929, 4.000)

#                 p(H)    p(M)        |   p(FA)
#                 0.519   0.481       |   0.296
                


#                                     -> 0.519, 0.296 -> 2.929
#                 (0.582, 0.244)    
#                                     -> 0.481, 0.296 -> 4.000

#         x = np.linspace(0, 1, 100)
#         coefs = poly.polyfit(fpr, tpr, 2) # Fit with polyfit
#         ffit = poly.polyval(x, coefs)
#         ax.plot(x, ffit, label='polyfit') # polyfit - fitting line with the dots.

#         """




def g(dprime, c):
    """ 1. function g(dprime, c) -> p(H), p(FA) """
    noi_d = scipy.stats.norm(loc=0, scale=1)
    sig_d = scipy.stats.norm(loc=dprime, scale=1) #where loc is the mean and scale is the std dev
    # estimated rates
    epm = sig_d.cdf(dprime/2 + c)
    epcr = noi_d.cdf(dprime/2 + c)
    eph = 1-epm # sig_d.sf(c)
    epfa = 1-epcr # noi_d.sf(c)
    return (eph, epfa)