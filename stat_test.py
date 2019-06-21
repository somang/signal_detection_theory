# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
import statsmodels.stats.power as smp
import pandas

import numpy
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

from xlrd import open_workbook
from xlsxwriter.workbook import Workbook

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

def export_excel(worksheet, all_list):
    for r in range(len(all_list)): # for each row
        row = all_list[r]
        for c in range(len(row)): # for each column
            worksheet.write(r,c,row[c]) # row, col, message

def to_score(answer):
    ansset_1 = ['1. Yes', '2. No']
    ansset_2 = ['1. Strongly disagree', '2. Disagree', '3. Neither agree nor disagree', '4. Agree', '5. Strongly agree']
    ansset_3 = ['1. Strongly dissatisfied', '2. Dissatisfied', '3. Neither satisfied nor dissatisfied', '4. Satisfied', '5. Strongly satisfied']
    ansset_4 = ['1. Very poor', '2. Poor', '3. Neither good nor poor', '4. Good', '5. Excellent']
    ans_cor = [ansset_1, ansset_2, ansset_3, ansset_4]

    for ansset in ans_cor:
        if answer in ansset:
            if answer == "2. No":
                return -1
            else:
                return int(answer[0])
def get_t_map(question, v1, v2):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x[1]), question[str(v2)].values()))
    return (v2, stats.ttest_ind(l1, l2))
    
def get_mannwhitneyu_map(question, v1, v2):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x[1]), question[str(v2)].values()))
    return (v2, mannwhitneyu(l1, l2))


def handle_satlvl(var_sat, x):
    return x * var_sat[x]





filename = "q1q4_results.xlsx"
xl = pandas.ExcelFile(filename)

workbook = Workbook('q1q4_handled.xlsx') # output
groups = {}
for sn in xl.sheet_names:
    variations = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {},
        "How would you rate your viewing pleasure from the video?": {}
    }
    df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)

    for i, r in df.iterrows():
        uuid, quiz, answer, caption = r[0], r[1], r[2], r[3]
        caption = caption.split("/")[1].split(".vtt")[0].split("_")        
        video, caption_var = caption[0], caption[2]
        # lets group by variations.
        if not variations[quiz].get(caption_var):
            variations[quiz][caption_var] = {}
        if not variations[quiz][caption_var].get(uuid):
            variations[quiz][caption_var][uuid] = (video, answer)

    all_list_video, all_list, uids = [], [], []
    for uid in variations["Did you see any errors in the caption?"]['1']:
        uids.append(uid)
       
    for q in variations:
        qz, qz_video = [], []
        head_row = ["uuid"]
        first = True
        for u in uids:
            row, vid_row = [u], [u]
            for n in range(1, 24): # from variation 1 to 23            
                variation_n = str(n) # convert to string value
                ans = (100, "") # initialization
                if first:
                    head_row.append(variation_n)
                if variations[q][variation_n].get(u):
                    ans = variations[q][variation_n][u]
                    row.append(to_score(ans[1]))
                    vid_row.append(ans[0])

            if first: # add the headers
                qz.append(head_row)
                qz_video.append(head_row)
                first = False
            qz.append(row) # for each row, it will have a username, along with the answers for each variation number 1-23
            qz_video.append(vid_row)

        all_list.append(qz) # add the row to the queue
        all_list_video.append(qz_video)

    groups[sn] = all_list

    worksheet = workbook.add_worksheet('q1_' + sn)
    export_excel(worksheet, all_list[0])
    worksheet = workbook.add_worksheet('q2_' + sn)
    export_excel(worksheet, all_list[1])
    worksheet = workbook.add_worksheet('q3_' + sn)
    export_excel(worksheet, all_list[2])
    worksheet = workbook.add_worksheet('q4_' + sn)
    export_excel(worksheet, all_list[3])
    worksheet = workbook.add_worksheet('v_' + sn)
    export_excel(worksheet, all_list_video[2])

workbook.close() # done writing

dprimes = {'all':{}, 'deaf':{}, 'hoh_deafened':{}}
ccq_satisfaction = {'all':{}, 'deaf':{}, 'hoh_deafened':{}}
viewing_pleasure = {'all':{}, 'deaf':{}, 'hoh_deafened':{}}

for g in groups:
    print(g)
    
    yn_freq, ccq, vp = {}, {}, {}
    for v in range(1,23):
        yn_freq[v] = {'y': 0, 'n': 0}    
        ccq[v] = {1: 0, 2: 0, 3:0, 4:0, 5:0}    
        vp[v] = {1: 0, 2: 0, 3:0, 4:0, 5:0}
    
    for uid in groups[g][0]: # get question
        for v in range(1,23):
            if uid[v] == 1:
                yn_freq[v]['y'] += 1
            elif uid[v] == -1:
                yn_freq[v]['n'] += 1

    # get SDT values
    tmp_l = []
    for v in range(2,23): # because variation 1 is for FA
        sdt_obj = SDT(HI=yn_freq[v]['y'], MI=yn_freq[v]['n'], FA=yn_freq[1]['y'], CR=yn_freq[1]['n'])
        tmp_l.append(sdt_obj.dprime())
    dprimes[g] = tmp_l

    # count the frequency of responses [1-5]
    for uid in groups[g][2]: # caption rating
        for v in range(1,23):
            if uid[v] == 1:
                ccq[v][1] += 1
            elif uid[v] == 2:
                ccq[v][2] += 1            
            elif uid[v] == 3:
                ccq[v][3] += 1
            elif uid[v] == 4:
                ccq[v][4] += 1
            elif uid[v] == 5:
                ccq[v][5] += 1
    ccq_satisfaction[g] = ccq

    # count the frequency of responses [1-5]
    for uid in groups[g][3]: # visual pleasure
        for v in range(1,23):
            if uid[v] == 1:
                vp[v][1] += 1
            elif uid[v] == 2:
                vp[v][2] += 1            
            elif uid[v] == 3:
                vp[v][3] += 1
            elif uid[v] == 4:
                vp[v][4] += 1
            elif uid[v] == 5:
                vp[v][5] += 1
    viewing_pleasure[g] = vp



#g = "deaf" 
g = "hoh_deafened"
cs = groups[g][2]
yn = groups[g][0]
yn_cr = groups[g][1]

rating = {'y':{}, 'n':{}}
for i in range(1, 23):
    rating['y'][i] = []
    rating['n'][i] = []


for user in range(1, len(groups[g][0])):
    uid = yn[user][0]
    for v in range(1,23):
        #print(uid + ":", yn[user][v] * yn_cr[user][v], cs[user][v])
        if yn[user][v] == 1:
            rating['y'][v].append(cs[user][v])
        elif yn[user][v] == -1:
            rating['n'][v].append(cs[user][v])

for v in range(1, 23):
    """
        #     #if v in [1, 6, 9, 14, 15, 20]:
        #     if v in [1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21]:
        #         print("variation {}: ({:.3f}, {:.3f})".format(v, numpy.mean(rating['y'][v]), numpy.mean(rating['n'][v])))
            # get mean rating
            # sum_v = 0
            # p_n = 0
            # for i in cs:
            #     p_n += cs[i]
            #     sum_v += i*cs[i]
            #     #print(i*cs[i], p_n)

            # print("{:.3f}".format(sum_v/p_n))
    """
    ### do t-test on quality ratings between yes-no, for each variation.
    y_shapiro = shapiro(rating['y'][v])
    n_shapiro = shapiro(rating['n'][v])
    if ((y_shapiro[1] > 0.05) and (n_shapiro[1] > 0.05)): # normal if p > .05
        print(v, "h(0): data are from normal distribution -> cannot be rejected. There's an evidence of normal distribution.")
        dhoh_levene = levene(rating['y'][v], rating['n'][v])
        print("Test for variance:", dhoh_levene)
        if dhoh_levene[1] > 0.05: # equal variance if p > .05
            print("h(0): data are from equal variances -> cannot be rejected. There's evidence of equal variance") 
            print("t-test assumptions are met")
            print(ttest_ind(rating['y'][v], rating['n'][v]))

    else:
        #print("variation{}:".format(v), len(rating['y'][v]), len(rating['n'][v]))
        mann_wit_result = mannwhitneyu(rating['y'][v], rating['n'][v])
        if mann_wit_result[1] < 0.05:
            print(v, "difference found", mann_wit_result)
        else:
            print("{} no significant difference - sampled from same distribution".format(v))

# for q in range(0,4):
#     print("question", q)

#     if q == 0:
#         print("deaf group mean, SD:", numpy.mean(dprimes['deaf']), numpy.std(dprimes['deaf']))
#         print("hoh group mean, SD:", numpy.mean(dprimes['hoh_deafened']), numpy.std(dprimes['hoh_deafened']))

#         d_shapiro = shapiro(dprimes['deaf'])
#         hoh_shapiro = shapiro(dprimes['hoh_deafened'])
#         print("Test for normality (deaf group, hoh group):", d_shapiro, hoh_shapiro)
#         if d_shapiro[1] > 0.05 and hoh_shapiro[1] > 0.05: # normal if p > .05
#             print("h(0): data are from normal distribution -> cannot be rejected. There's an evidence of normal distribution.")            
#             dhoh_levene = levene(dprimes['hoh_deafened'], dprimes['deaf'])
#             print("Test for variance:", dhoh_levene)
#             if dhoh_levene[1] > 0.05: # equal variance if p > .05
#                 print("h(0): data are from equal variances -> cannot be rejected. There's evidence of equal variance") 
#                 print("t-test assumptions are met")
#                 df = len(dprimes['deaf']) + len(dprimes['hoh_deafened']) - 2
#                 #print(ttest_ind(dprimes['deaf'], dprimes['hoh_deafened']), 'df:'+str(df))
#                 print(ttest_ind(dprimes['hoh_deafened'], dprimes['deaf']), 'df:'+str(df))

#                 print(len(dprimes['deaf']), len(dprimes['hoh_deafened']))
#                 print(smp.ttest_power(0.8, nobs=len(dprimes['deaf']), alpha=0.05, alternative='two-sided'))
#                 print(smp.ttest_power(0.8, nobs=len(dprimes['hoh_deafened']), alpha=0.05, alternative='two-sided'))
                
#                 # parameters for power analysis
#                 effect = 0.8
#                 alpha = 0.05
#                 power = 0.8
#                 # perform power analysis
#                 result = TTestIndPower().solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
#                 print('Sample Size: %.3f' % result) # but we have only 21 sensitivities for each group to compare...

#             else:
#                 print("not equal variance........")
#         else:
#             print("probable not normally distributed")

#             print(mannwhitneyu(dprimes['deaf'], dprimes['hoh_deafened']))



        ### Find the difference between main effect vs interaction effect.        
        # for v in range(21):
        #     print(v, dprimes['deaf'][v], dprimes['hoh_deafened'][v])
        # d_main, d_int = dprimes['deaf'][:8], dprimes['deaf'][8:]
        # h_main, h_int = dprimes['hoh_deafened'][:8], dprimes['hoh_deafened'][8:]
        # print(ttest_ind(d_main, d_int))
        # print(ttest_ind(h_main, h_int))
        ######## No significant difference between main vs interaction variations (even with mann-whitney)




    # if q == 2:
    #     d_qr = []
    #     h_qr = []
    #     for v in range(1, 23):
    #         # var_sat = ccq_satisfaction['all'][v]
    #         # tmp = []
    #         # for i in var_sat:
    #         #     for j in range(var_sat[i]):
    #         #         tmp.append(i)
    #         # print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp), numpy.std(tmp), numpy.median(tmp)))

    #         d_ccq = ccq_satisfaction['deaf'][v]
    #         tmp_d = []
    #         for i in d_ccq:
    #             for j in range(d_ccq[i]):
    #                 tmp_d.append(i)
    #         #print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp_d), numpy.std(tmp_d), numpy.median(tmp_d)))

    #         h_ccq = ccq_satisfaction['hoh_deafened'][v]
    #         tmp_h = []
    #         for i in h_ccq:
    #             for j in range(h_ccq[i]):
    #                 tmp_h.append(i)
    #         #print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp_h), numpy.std(tmp_h), numpy.median(tmp_h)))

    #         d_qr.append(numpy.mean(tmp_d))
    #         h_qr.append(numpy.mean(tmp_h))
        
    #     # see if there's difference in ranked between deaf and hoh viewers mean quality rating
    #     print("deaf group mean, SD:", numpy.mean(d_qr), numpy.std(d_qr))
    #     print("hoh group mean, SD:", numpy.mean(h_qr), numpy.std(h_qr))

    #     d_shapiro = shapiro(d_qr)
    #     hoh_shapiro = shapiro(h_qr)
    #     print("Test for normality (deaf group, hoh group):", d_shapiro, hoh_shapiro)

    #     if d_shapiro[1] > 0.05 and hoh_shapiro[1] > 0.05: # normal if p > .05
    #         print("h(0): data are from normal distribution -> cannot be rejected. There's an evidence of normal distribution.")
    #         dhoh_levene = levene(d_qr, h_qr)
    #         print("Test for variance:", dhoh_levene)
    #         if dhoh_levene[1] > 0.05: # equal variance if p > .05
    #             print("h(0): data from equal variances -> cannot be rejected. There's evidence of equal variance") 
    #             print("t-test assumptions are met")
    #             df = len(d_qr) + len(h_qr) - 2
    #             #print(ttest_ind(d_qr, h_qr), 'df:'+str(df))
    #             print(ttest_ind(h_qr, d_qr), 'df:'+str(df))
    #         else:
    #             print("not equal variance........")
    #     else:
    #         print("probable not normally distributed")
    #         print(mannwhitneyu(d_qr, h_qr))
    # if q == 3:
    #     d_qr = []
    #     h_qr = []
    #     for v in range(1, 23):
    #         # var_sat = viewing_pleasure['all'][v]
    #         # tmp = []
    #         # for i in var_sat:
    #         #     for j in range(var_sat[i]):
    #         #         tmp.append(i)
    #         # print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp), numpy.std(tmp), numpy.median(tmp)))

    #         d_ccq = viewing_pleasure['deaf'][v]
    #         tmp_d = []
    #         for i in d_ccq:
    #             for j in range(d_ccq[i]):
    #                 tmp_d.append(i)
    #         #print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp_d), numpy.std(tmp_d), numpy.median(tmp_d)))

    #         h_ccq = viewing_pleasure['hoh_deafened'][v]
    #         tmp_h = []
    #         for i in h_ccq:
    #             for j in range(h_ccq[i]):
    #                 tmp_h.append(i)
    #         #print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp_h), numpy.std(tmp_h), numpy.median(tmp_h)))

    #         d_qr.append(numpy.mean(tmp_d))
    #         h_qr.append(numpy.mean(tmp_h))
        
    #     # see if there's difference in ranked between deaf and hoh viewers mean quality rating
    #     print("deaf group mean, SD:", numpy.mean(d_qr), numpy.std(d_qr))
    #     print("hoh group mean, SD:", numpy.mean(h_qr), numpy.std(h_qr))

    #     d_shapiro = shapiro(d_qr)
    #     hoh_shapiro = shapiro(h_qr)
    #     print("Test for normality (deaf group, hoh group):", d_shapiro, hoh_shapiro)

    #     if d_shapiro[1] > 0.05 and hoh_shapiro[1] > 0.05: # normal if p > .05
    #         print("h(0): data are from normal distribution -> cannot be rejected. There's an evidence of normal distribution.")
    #         dhoh_levene = levene(d_qr, h_qr)
    #         print("Test for variance:", dhoh_levene)
    #         if dhoh_levene[1] > 0.05: # equal variance if p > .05
    #             print("h(0): data from equal variances -> cannot be rejected. There's evidence of equal variance") 
    #             print("t-test assumptions are met")
    #             df = len(d_qr) + len(h_qr) - 2
    #             #print(ttest_ind(d_qr, h_qr), 'df:'+str(df))
    #             print(ttest_ind(h_qr, d_qr), 'df:'+str(df))
    #         else:
    #             print("not equal variance........")
    #     else:
    #         print("probable not normally distributed")
    #         print(mannwhitneyu(d_qr, h_qr))