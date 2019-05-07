# let's read q1q3handled
import csv
from xlsxwriter.workbook import Workbook

import xlrd
from xlrd import open_workbook

import pandas

from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats.stats import pearsonr   

import math


def multiply_two(x):
    yn = x[0] 

    if yn == 2:
        yn = -1
    return yn*x[1]



filename = 'q1q3_handled.xlsx'
xl = pandas.ExcelFile(filename)

# q1 "Did you see any errors in the caption?"
# q2 "I am confident that my decision was correct:"
# q3 "How would you rate the quality of the caption?"

yns, confs, ratings = {}, {}, {}
for v in range(1,24):
    yns[v], confs[v], ratings[v] = [], [], []

for sn in xl.sheet_names:

    if sn == "q1_all":
        df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)
        
        for i, r in df.iterrows():
            if not r[0] == "uuid":
                for c in range(1,24):
                    if not math.isnan(r[c]):
                        yns[c].append(r[c])
            
    elif sn == "q2_all":
        df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)
        
        for i, r in df.iterrows():
            if not r[0] == "uuid":
                for c in range(1,24):
                    if not math.isnan(r[c]):
                        confs[c].append(r[c])

    elif sn == "q3_all":
        df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)
        
        for i, r in df.iterrows():
            if not r[0] == "uuid":
                for c in range(1,24):
                    if not math.isnan(r[c]):
                        ratings[c].append(r[c])


for v in range(1,24):
    p = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))
    # p = yns multiply by conf
    # check pearson's r

    print(v, pearsonr(p, ratings[v]))

"""
            (Pearson's r, and p value)

    1 (-0.6903991046957001, 2.0749428461611953e-08)
    2 (-0.7918886473618019, 4.522708727242528e-12)
    3 (-0.7488112608240346, 2.6364397473234106e-10)
    4 (-0.6064172065606668, 2.399698291620623e-06)
    5 (-0.7170116189850018, 3.2482996690650618e-09)
    6 (-0.728299284752957, 1.3866161063091653e-09)
    7 (-0.6979540886471706, 1.250871016311431e-08)
    8 (-0.6381019697477913, 4.729387557692429e-07)
    9 (-0.7460925720531371, 3.3149987388718865e-10)
    10 (-0.6604038999985622, 1.3434755631663117e-07)
    11 (-0.8309331064502419, 4.510558699523016e-14)
    12 (-0.7162881105404091, 3.425689867315723e-09)
    13 (-0.8412343501383541, 1.0963897306672743e-14)
    14 (-0.7523320135036176, 1.951284516537525e-10)
    15 (-0.7310351436126618, 1.1209477935382006e-09)
    16 (-0.6311624164102753, 6.856039732132431e-07)
    17 (-0.7162491774858943, 3.4354896671924548e-09)
    18 (-0.6771755322608088, 4.855419748347722e-08)
    19 (-0.7744017509572155, 2.6220555671158666e-11)
    20 (-0.6325111982053204, 6.383225804677322e-07)
    21 (-0.6504299614023958, 2.388876293302704e-07)
    22 (-0.8280642973192835, 6.576563641419638e-14)
    23 (-0.8208664600390161, 0.0001764667725424937)
"""