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

xl = pandas.ExcelFile('q1q3_handled.xlsx')



def multiply_two(x):
    yn = x[0] # check the y/n answer
    if yn == 2: # if the answer is no (2. No)
        yn = -1 # make it negative.
    return yn*x[1]

def handle_list(df, di):
    for i, r in df.iterrows():
        if not r[0] == "uuid":
            for c in range(1,24):
                if not math.isnan(r[c]):
                    di[c].append(r[c])
    return di

def print_pearson(yns, confs, ratings):
    for v in range(1,24):
        # p = yns multiply by conf then check pearson's r
        p = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))
        print(v, pearsonr(p, ratings[v])) # x, y



# q1 "Did you see any errors in the caption?"
# q2 "I am confident that my decision was correct:"
# q3 "How would you rate the quality of the caption?"

yns, confs, ratings = {}, {}, {}
d_yns, d_confs, d_ratings = {}, {}, {}
hoh_yns, hoh_confs, hoh_ratings = {}, {}, {}
for v in range(1,24):
    yns[v], confs[v], ratings[v] = [], [], []
    d_yns[v], d_confs[v], d_ratings[v] = [], [], []
    hoh_yns[v], hoh_confs[v], hoh_ratings[v] = [], [], []

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

    elif sn == "q1_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_yns = handle_list(df, d_yns)
    elif sn == "q2_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_confs = handle_list(df, d_confs)
    elif sn == "q3_deaf":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        d_ratings = handle_list(df, d_ratings)

    elif sn == "q1_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_yns = handle_list(df, hoh_yns)
    elif sn == "q2_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_confs = handle_list(df, hoh_confs)
    elif sn == "q3_hoh_deafened":
        df = pandas.read_excel(xl, sheet_name=sn, header=None, index_col=False)        
        hoh_ratings = handle_list(df, hoh_ratings)


print_pearson(yns, confs, ratings)
print()
print_pearson(d_yns, d_confs, d_ratings)
print()
print_pearson(hoh_yns, hoh_confs, hoh_ratings)

"""
    (Pearson's r, and p value) 0.6 or greater is the strong correlation

    y at highest is 5. Strongly satisfied
    x converted range is now from -5 to +5, where            
    negative represents 'no' answers with the strongest confidence at 5 and
    positive represents 'yes' answers with the strongest confidence at 5
    thus,
        (-5 = strongly no error) to (+5= strongly yes error)

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

    Deaf Group
    1 (-0.6791488714843478, 0.0001892085373498975)
    2 (-0.7793777505014189, 4.405866859702371e-06)
    3 (-0.7241641018637195, 4.266194523973745e-05)
    4 (-0.7657229119159616, 8.170316710669746e-06)
    5 (-0.6732080678399429, 0.00022601071189125313)
    6 (-0.8226271156731979, 4.5021330451848695e-07)
    7 (-0.7044228478306448, 8.475115686357541e-05)
    8 (-0.6784307641851369, 0.00019335835358366796)
    9 (-0.811901253751343, 8.360789831366738e-07)
*10 (-0.5000216103680054, 0.01091850765813249)
    11 (-0.7710362606718106, 6.456977371677826e-06)
    12 (-0.7414985531765919, 2.2228038319346734e-05)
    13 (-0.8260152596860131, 3.6709466295228194e-07)
    14 (-0.8073313867530838, 1.0757237157846836e-06)
    15 (-0.7871335397431852, 3.042453224898414e-06)
    16 (-0.653891573970952, 0.00039238234547147303)
    17 (-0.6685608869296673, 0.00025901625339385824)
    18 (-0.627015630553054, 0.0007957036986709799)
    19 (-0.7006817149043738, 9.593622173261442e-05)
    20 (-0.718985373137958, 5.1359314193767324e-05)
    21 (-0.6776650443120572, 0.00019787083019121935)
    22 (-0.7960445750006154, 1.950578593351173e-06)
    23 (-0.8780689277123096, 0.0008325918098380031)

    Hoh Group
    1 (-0.6827978110237571, 0.00012132495681738696)
    2 (-0.8129187787454173, 4.5076713307458594e-07)
    3 (-0.7914337708679904, 1.4748511329182255e-06)
*4 (-0.4141591213263059, 0.03542619458754293)
    5 (-0.7797571855624496, 2.656368881529984e-06)
*6 (-0.5604954560910205, 0.002899185626548467)
    7 (-0.6208017909258409, 0.0007143094950074238)
    8 (-0.6025113269059399, 0.0011249361638735829)
    9 (-0.6490547893968336, 0.000334248562220935)
    10 (-0.8145649566715623, 4.091070829053028e-07)
    11 (-0.9116376899567672, 9.4672768546475e-11)
    12 (-0.7379240156466277, 1.6894062536711108e-05)
    13 (-0.8612627431333959, 1.623825646050812e-08)
    14 (-0.6898666808020799, 9.649087169026114e-05)
    15 (-0.6816564196320879, 0.00012582264894398938)
    16 (-0.6103812069381422, 0.0009283837342641645)
    17 (-0.7827952338336946, 2.287132938434615e-06)
    18 (-0.7292586522065966, 2.3752065777248956e-05)
    19 (-0.838507673216582, 8.879830483118291e-08)
*20 (-0.5639277193448823, 0.002695853740575549)
    21 (-0.6075177456706118, 0.0009961476396654415)
    22 (-0.8535030908880105, 2.991813441358625e-08)
    23 (-0.6814501278449928, 0.205198408773597)


    1. Write a set of hypothesis
        - In general having multiple errors can be more detectable
        - stronger main effects also in the interaction effects.
        - 

    - Paraphrasing in translation:
        Subtitling

    - Psychology of word recognition
        - Missing words
    - Error detection in language
    - Reading speed
    - Closed Captioning
        - Delay
        - Edited captions
    - Signal detection theory



    


"""