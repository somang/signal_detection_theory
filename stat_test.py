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

for q in range(0,4):
    print("question", q)
    if q == 0:
        print(numpy.mean(dprimes['deaf']), numpy.std(dprimes['deaf']))
        print(numpy.mean(dprimes['hoh_deafened']), numpy.std(dprimes['hoh_deafened']))

        d_shapiro = shapiro(dprimes['deaf'])
        hoh_shapiro = shapiro(dprimes['hoh_deafened'])
        print("Test for normality:", d_shapiro, hoh_shapiro)
        if d_shapiro[1] > 0.05 and hoh_shapiro[1] > 0.05: # normal if p > .05
            print("There's an evidence of normal distribution.")
            dhoh_levene = levene(dprimes['deaf'], dprimes['hoh_deafened'])
            print("Test for variance:", dhoh_levene)
            if dhoh_levene[1] > 0.05: # equal variance if p > .05
                print("There's evidence of equal variance") 
                print("t-test assumptions are met")
                df = len(dprimes['deaf']) + len(dprimes['hoh_deafened']) - 2
                print(ttest_ind(dprimes['deaf'], dprimes['hoh_deafened']), 'df:'+str(df))
            else:
                print("not equal variance........")
        else:
            print("probable not normally distributed")
            print(mannwhitneyu(dprimes['deaf'], dprimes['hoh_deafened']))
    if q == 2:
        d_ccq = ccq_satisfaction['deaf']
        h_ccq = ccq_satisfaction['hoh_deafened']
        for v in range(1, 23):
            var_sat = ccq_satisfaction['all'][v]
            tmp = []
            for i in var_sat:
                for j in range(var_sat[i]):
                    tmp.append(i)
            # tmp has all occurences of votes/ratings
            print("variation #{0:d}: mean={1:.3f}, sd={2:.3f}, median={3:.3f}".format(v, numpy.mean(tmp), numpy.std(tmp), numpy.median(tmp)))




            # list(map(lambda x: handle_satlvl(var_sat, x), var_sat))            
