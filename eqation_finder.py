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

def print_pearson(yns, confs, ratings):
    for v in range(1,24):
        # p = yns multiply by conf then check pearson's r
        p = list(map(lambda x: multiply_two(x), list(zip(yns[v], confs[v]))))
        print(v, pearsonr(p, ratings[v])) # x, y



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

print("y/n * confidence  VS caption quality ratings")
print_pearson(yns, confs, ratings)
print()
print_pearson(d_yns, d_confs, d_ratings)
print()
print_pearson(hoh_yns, hoh_confs, hoh_ratings)

print("y/n * confidence  VS visual pleasure ratings")
print_pearson(yns, confs, vpr)
print()
print_pearson(d_yns, d_confs, d_vpr)
print()
print_pearson(hoh_yns, hoh_confs, hoh_vpr)



"""
    (Pearson's r, and p value) 0.6 or greater is the strong correlation

    y at highest is 5. Strongly satisfied
    x converted range is now from -5 to +5, where            
    negative represents 'no' answers with the strongest confidence at 5 and
    positive represents 'yes' answers with the strongest confidence at 5
    thus,
        (-5 = strongly no error) to (+5= strongly yes error)

    y/n * confidence  VS caption quality ratings         y/n * confidence  VS visual pleasure ratings
    1 (0.6934116720555364, 1.2123793570989079e-08)       1 (0.6994197293341081, 8.011935220857568e-09)
    2 (0.7782973126651795, 1.1174795316256021e-11)       2 (0.7644486323939246, 4.251524884354351e-11)
    3 (0.7386577587759187, 4.0806230222498933e-10)       3 (0.7084939125718878, 4.20277516365906e-09)
    4 (0.6228990812126216, 8.170788153057826e-07)        4 (0.6626916779029673, 8.706446113181291e-08)
    5 (0.722357321606623, 1.4942170756334945e-09)        5 (0.6324413761709398, 4.915948329118226e-07)
    6 (0.71479799197555, 2.6458784855138356e-09)         6 (0.7656029863583972, 3.8167340033422367e-11)
    7 (0.6996298856243197, 7.895236393028618e-09)        7 (0.6493892086737734, 1.9084618385290264e-07)
    8 (0.6139175485029406, 1.2981442640246666e-06)       *8 (0.5042386611449238, 0.0001382147671356377)
    9 (0.7494186999125019, 1.6417992207476572e-10)       9 (0.7132507844426098, 2.967470215723197e-09)
    10 (0.6242044896636423, 7.629807296959355e-07)      10 (0.6837456805463639, 2.3128877217244147e-08) 
    11 (0.8326291530542611, 1.972738117165213e-14)     11 (0.7863315168072409, 4.922940496758584e-12)
    12 (0.7232596704381861, 1.3939617817140637e-09)    12 (0.7090336457198602, 4.041445612612989e-09)
    13 (0.8443437422649699, 3.721111425532054e-15)     13 (0.7183468412648231, 2.0279879820735865e-09)
    14 (0.7203850371710122, 1.7375390232497328e-09)    14 (0.7121330031783312, 3.2223667856792145e-09)
    15 (0.701949630358653, 6.70907857797579e-09)       15 (0.6042622867145245, 2.101941460450887e-06)
    16 (0.6050665852385905, 2.020452964978506e-06)     *16 (0.530779382584689, 5.1687263181954564e-05)
    17 (0.6790631855689968, 3.135126107194862e-08)     17 (0.6000170288991284, 2.585133851504098e-06)
    18 (0.6461637765336933, 2.295312265392969e-07)     18 (0.7086622309937275, 4.151823664696164e-09)
    19 (0.7823096860487081, 7.452695255798086e-12)     19 (0.8107341923565459, 3.238337228946914e-13)
    20 (0.6043553932301777, 2.0923541254994182e-06)    20 (0.6300364198558375, 5.596578453797865e-07)
    21 (0.6545008688743256, 1.4180516134685803e-07)    *21 (0.5521458234670109, 2.2008777054213438e-05)
    22 (0.8317963999301691, 2.2102066934715805e-14)    22 (0.7387051432957716, 4.0646915222840155e-10)

y/n * confidence  VS caption quality ratings deaf       y/n * confidence  VS visual pleasure ratings deaf
    1 (0.6791488714843478, 0.0001892085373498975)    1 (0.6843425108353508, 0.00016144550605519452)
    2 (0.7793777505014189, 4.405866859702371e-06)    2 (0.8073243760287109, 1.076134235378091e-06)
    3 (0.7241641018637195, 4.266194523973745e-05)    3 (0.7068708605471281, 7.806869692037113e-05)
    4 (0.7657229119159616, 8.170316710669746e-06)    4 (0.6922106276671355, 0.00012617497786219085)
    5 (0.6732080678399429, 0.00022601071189125313)   5 (0.7051525201144425, 8.270861975520123e-05)
    6 (0.8226271156731979, 4.5021330451848695e-07)   6 (0.7807658480874781, 4.127830326484747e-06)
    7 (0.7044228478306448, 8.475115686357541e-05)    7 (0.6201102516235264, 0.0009443988448845036)
    8 (0.6784307641851369, 0.00019335835358366796)   *8 (0.5266655253861335, 0.006835486548555502)
    9 (0.811901253751343, 8.360789831366738e-07)     9 (0.6560994914173244, 0.0003691208843694338)
    *10 (0.5000216103680054, 0.01091850765813249)   *10 (0.573268566944182, 0.0027387551099232643)
    11 (0.7710362606718106, 6.456977371677826e-06)  11 (0.7069760526704084, 7.779224674489678e-05)
    12 (0.7414985531765919, 2.2228038319346734e-05) 12 (0.7073600179272368, 7.679046361868264e-05)
    13 (0.8260152596860131, 3.6709466295228194e-07) 13 (0.6937422042803477, 0.00012015784278650258)
    14 (0.8073313867530838, 1.0757237157846836e-06) 14 (0.7539684367089765, 1.3467111990630453e-05)
    15 (0.7871335397431852, 3.042453224898414e-06)  *15 (0.5954074368373726, 0.0016894226769362123)
    16 (0.653891573970952, 0.00039238234547147303)  *16 (0.5288712961974031, 0.006564533909830405)
    17 (0.6685608869296673, 0.00025901625339385824) 17 (0.6786761568194788, 0.00019193140498151718)
    18 (0.627015630553054, 0.0007957036986709799)   18 (0.6843966097119601, 0.00016117620135701476)
    19 (0.7006817149043738, 9.593622173261442e-05)  19 (0.7691213883393245, 7.03350167516757e-06)
    20 (0.718985373137958, 5.1359314193767324e-05)  20 (0.7726370774130603, 6.007692675416917e-06)
    21 (0.6776650443120572, 0.00019787083019121935) *21 (0.5781450587340454, 0.0024694208492522787)
    22 (0.7960445750006154, 1.950578593351173e-06)  22 (0.7269150241831095, 3.859344445927583e-05)
    
    
y/n * confidence  VS caption quality ratings hoh        y/n * confidence  VS visual pleasure ratings hoh
    1 (0.6856953333067284, 7.897351054242847e-05)       1 (0.7035820971432658, 4.236262150848809e-05)
    2 (0.777308379580687, 1.846566109803697e-06)        2 (0.6944166082624115, 5.860962432946693e-05)
    3 (0.7789427486763594, 1.700413063168166e-06)       3 (0.7417473804517274, 9.52595855773058e-06)
    4 (0.46762967298218927, 0.013909532230112893)       4 (0.6379385433994547, 0.00034379434041255196)
    5 (0.7927372183621376, 8.239507311635558e-07)       *5 (0.5918226127403504, 0.001147093940443039)    
    *6 (0.5284800155374871, 0.004599948085299518)       6 (0.7095662904602298, 3.4048104467097946e-05)
    7 (0.622701223234057, 0.0005227188600656141)        *7 (0.5996570664502331, 0.0009468037805811635)
    *8 (0.5575758468904136, 0.0025140080370970974)      *8 (0.505920238351375, 0.007095362583659542)
    9 (0.6551180332039734, 0.0002085349513223972)       9 (0.756577919312706, 4.9692008278600446e-06)
    10 (0.7531284604584093, 5.804416754663841e-06)      10 (0.792278404493432, 8.447733442382973e-07)
    11 (0.9105737813302638, 4.404669664383666e-11)      11 (0.8785610350606673, 1.6885290534353994e-09)
    12 (0.7603514670489547, 4.180313250415423e-06)      12 (0.7407349146869462, 9.942966677579454e-06)
    13 (0.8663608656594878, 5.2160748017660125e-09)     13 (0.751032414494889, 6.371341746961277e-06)
    14 (0.639591397553554, 0.0003280779146684261)       14 (0.658621089382445, 0.00018760604393228274)
    15 (0.6345481732324338, 0.00037811049959856)        15 (0.6109635504183665, 0.0007114610505441194)
    *16 (0.5703672443451349, 0.001893973251081301)      *16 (0.5349704425304049, 0.004038437297954497)
    17 (0.7231787138479309, 2.0292426281514727e-05)     *17 (0.5363911350482593, 0.003923640546707289)
    18 (0.6614511885120283, 0.00017207186991566)        18 (0.7265582436271445, 1.776280651092359e-05)
    19 (0.8515703198667196, 1.7813910176868792e-08)     19 (0.8494293641612222, 2.10479896939406e-08)
    *20 (0.522188607632953, 0.005206172719598212)       *20 (0.48837917046209933, 0.0097482638215175)
    21 (0.6144836969526927, 0.0006494590139605332)      *21 (0.5079939762735972, 0.0068264125734044615)
    22 (0.865613876131843, 5.56890541775472e-09)        22 (0.7179000857088507, 2.488908031325818e-05)
    


    1. Write a set of hypothesis
        - In general having multiple errors can be more detectable
        - stronger main effects also in the interaction effects.
    


"""