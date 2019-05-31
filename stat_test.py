import csv
from xlsxwriter.workbook import Workbook
import xlrd
from xlrd import open_workbook
import pandas
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import sdt_metrics
from sdt_metrics import dprime, HI, MI, CR, FA, SDT

from scipy import stats


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

def get_shapiro_map(question, v1):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    return (v1, shapiro(l1))


filename = "q1q4_results.xlsx"
xl = pandas.ExcelFile(filename)

workbook = Workbook('q1q4_handled.xlsx') # output
groups = {}
for sn in xl.sheet_names:
    #print(sn)
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
    for uid in variations["Did you see any errors in the caption?"]['1']: # get all uids
        uids.append(uid)
       
    for q in variations: # for each question
        qz, qz_video = [], []
        head_row = ["uuid"]
        first = True
        for u in uids: # for each user
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
for g in groups:
    print(g)
    tmp_dprime = []
    # count yes/no for each variation
    gq = groups[g][0] # yes/no question
    freq = {}
    for v in range(1,23):
        freq[v] = {'y': 0, 'n': 0}    
    for uid in gq:
        for v in range(1,23):
            if uid[v] == 1:
                freq[v]['y'] += 1
            elif uid[v] == -1:
                freq[v]['n'] += 1
    # get SDT values
    tmp_l = []
    for v in range(2,23): # because variation 1 is for FA
        sdt_obj = SDT(HI=freq[v]['y'], MI=freq[v]['n'], FA=freq[1]['y'], CR=freq[1]['n'])
        tmp_l.append(sdt_obj.dprime())
    dprimes[g] = tmp_l


for q in range(0,3):
    print("question", q)
    if q == 0:
        print(mannwhitneyu(dprimes['deaf'], dprimes['hoh_deafened']))
        print(stats.ttest_ind(dprimes['deaf'], dprimes['hoh_deafened']))
    else:
        print(groups['deaf'][1])
        print(mannwhitneyu(groups['deaf'][q], groups['hoh_deafened'][q]))
        print(stats.ttest_ind(groups['deaf'][q], groups['hoh_deafened'][q]))