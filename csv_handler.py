import csv
from xlsxwriter.workbook import Workbook
import xlrd
from xlrd import open_workbook
import pandas
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

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
        if ans in ansset:
            if ans == "2. No":
                return -1
            else:
                return int(ans[0])

def get_weight_score(x):
    if x[1] != None:
        return float(x[0]) * float(x[1])
    
def get_t_map(question, v1, v2):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x[1]), question[str(v2)].values()))
    return (v2, stats.ttest_ind(l1, l2))
    
def get_shapiro_map(question, v1):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    return (v1, shapiro(l1))

def get_mannwhitneyu_map(question, v1, v2):
    l1 = list(map(lambda x: to_score(x[1]), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x[1]), question[str(v2)].values()))
    return (v2, mannwhitneyu(l1, l2))

workbook = Workbook('q1q3_handled.xlsx') # output
filename = "q1q3_results.xlsx"
xl = pandas.ExcelFile(filename)
"""
Is narrator in the video clip?
[v2_sports: no, v3_sports: no, v4_hockey: no, v5_hockey: no, v6_hockey: no, 
v7_weather: yes, but cannot see face, v8_weather: yes, but cannot see lips clearly, v9_weather: yes, but cannot see face]
[v1_sports: yes, v10_social: yes, v11_social: yes, v12_social: yes, v13_weather: yes, v14_weather: yes, v15_weather: yes,
v16_weather: yes, v17_weather: yes, v18_weather: yes, v19_weather: yes, v20_weather: yes, v21_weather: yes, v22_weather: yes
v23_social: yes]
"""


narrator_vid = ["v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"]
no_narrator_vid = ["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]


for sn in xl.sheet_names:
    print(sn)

    df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)

    variations = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {},
        "How would you rate your viewing pleasure from the video?": {}
    }

    nar = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {},
        "How would you rate your viewing pleasure from the video?": {}
    }

    no_nar = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {},
        "How would you rate your viewing pleasure from the video?": {}
    }

    for i, r in df.iterrows():
        uuid, quiz, answer, caption = r[0], r[1], r[2], r[3]
        caption = caption.split("/")[1].split(".vtt")[0].split("_")
        #video, caption_var = "_".join(caption[:2]), caption[2]
        video, caption_var = caption[0], caption[2]

        if video in narrator_vid:
            if not nar[quiz].get(caption_var):
                nar[quiz][caption_var] = {}
            if not nar[quiz][caption_var].get(uuid):
                nar[quiz][caption_var][uuid] = (video, answer)

        elif video in no_narrator_vid:
            if not no_nar[quiz].get(caption_var):
                no_nar[quiz][caption_var] = {}
            if not no_nar[quiz][caption_var].get(uuid):
                no_nar[quiz][caption_var][uuid] = (video, answer)

        # lets group by variations.
        if not variations[quiz].get(caption_var):
            variations[quiz][caption_var] = {}
        if not variations[quiz][caption_var].get(uuid):
            variations[quiz][caption_var][uuid] = (video, answer)

    ## lets calculate t-tests over the columns we want,
    for q in variations: #variations:
        qmap = variations[q]
        
        shapiro_result = list(map(lambda x: get_shapiro_map(qmap, x), range(1, 24))) # check for normality
        for s in shapiro_result:
            p = s[1][1]
            # interpret
            alpha = 0.05
            if p > alpha:
                print(s, 'Sample looks Gaussian (fail to reject H0)')
            # else: # p<.05, there's evidence that the data are not normally distributed
            #     print(s, 'Sample does not look Gaussian (reject H0)')        
        
        mwu_result = list(map(lambda x: get_mannwhitneyu_map(qmap, 1, x), range(1, 24)))
        for s in mwu_result:
            p = s[1][1]
            # interpret
            alpha = 0.05
            if p < alpha: # p<.05, evidence that two independent samples were NOT drawn from a population with the same distribution
                print(s, 'Different distribution (reject H0)')
            # else: 
            #     print(s, 'Same distribution (fail to reject H0)')
        
        ttest_result = list(map(lambda x: get_t_map(qmap, 1, x), range(2,24))) # ttest( v1 , (v2-v23) )
        for s in ttest_result:
            print(s)

            p = s[1][1]
            # interpret
            alpha = 0.05
            # interpret via p-value
            # if p < alpha:
            #     print(s, 'Reject the null hypothesis that the means are equal.')        
            # else:
            #     print(s, 'Accept null hypothesis that the means are equal.')

    























    # now we have a sorted table.
    all_list_video, all_list, nar_list, no_nar_list, uids = [], [], [], [], []
    for uid in variations["Did you see any errors in the caption?"]['1']: # get all uids
        uids.append(uid)
    
    for q in variations: # for each question
        qz, qz_video, q_nar, q_nonar = [], [], [], []
        head_row = ["uuid"]
        first = True
        for u in uids: # for each user
            row, vid_row, r_nar, r_nonar = [u], [u], [u], [u]
            
            for n in range(1, 24): # from variation 1 to 23            
                variation_n = str(n) # convert to string value
                ans = (100, "") # initialization
                if first:
                    head_row.append(variation_n)

                if nar[q][variation_n].get(u):
                    ans = nar[q][variation_n][u] # ('v15', '1. Strongly dissatisfied')
                    r_nar.append(to_score(ans[1]))
                else:
                    r_nar.append(" ")

                if no_nar[q][variation_n].get(u):
                    ans = no_nar[q][variation_n][u]
                    r_nonar.append(to_score(ans[1]))
                else:
                    r_nonar.append(" ") # pad with a dummy number if not existing



                if variations[q][variation_n].get(u):
                    ans = variations[q][variation_n][u]
                    row.append(to_score(ans[1]))
                    vid_row.append(ans[0])

            if first: # add the headers
                qz.append(head_row)
                qz_video.append(head_row)
                q_nar.append(head_row)
                q_nonar.append(head_row)
                first = False

            qz.append(row) # for each row, it will have a username, along with the answers for each variation number 1-23
            qz_video.append(vid_row)
            q_nar.append(r_nar)
            q_nonar.append(r_nonar)

        all_list.append(qz)# add the row to the queue
        all_list_video.append(qz_video)
        nar_list.append(q_nar)
        no_nar_list.append(q_nonar)


    # worksheet = workbook.add_worksheet('q1_' + sn)
    # export_excel(worksheet, all_list[0])
    # worksheet = workbook.add_worksheet('q2_' + sn)
    # export_excel(worksheet, all_list[1])
    # worksheet = workbook.add_worksheet('q3_' + sn)
    # export_excel(worksheet, all_list[2])
    # worksheet = workbook.add_worksheet('v_' + sn)
    # export_excel(worksheet, all_list_video[2])


    # worksheet = workbook.add_worksheet('nar_q1' + sn)
    # export_excel(worksheet, nar_list[0])
    # worksheet = workbook.add_worksheet('nar_q2' + sn)
    # export_excel(worksheet, nar_list[1])
    # worksheet = workbook.add_worksheet('nar_q3' + sn)
    # export_excel(worksheet, nar_list[2])


    # worksheet = workbook.add_worksheet('nonar_q1' + sn)
    # export_excel(worksheet, no_nar_list[0])
    # worksheet = workbook.add_worksheet('nonar_q2' + sn)
    # export_excel(worksheet, no_nar_list[1])
    # worksheet = workbook.add_worksheet('nonar_q3' + sn)
    # export_excel(worksheet, no_nar_list[2])


# #     # now that we have the raw values, let's calculate the weighted scores.
# #     q4 = []
# #     for r1 in all_list[0]:
# #         for r2 in all_list[1]:
# #             if (r1[0] == r2[0]) and (r1[0] != "uuid"):
# #                 new_val_row = r1[:1] + list(map(lambda x: get_weight_score(x), zip(r1[1:], r2[1:])))
# #                 q4.append(new_val_row)
# #     all_list.append(q4)            

#     # worksheet = workbook.add_worksheet('q4_' + sn)
#     # export_excel(worksheet, all_list[3])

workbook.close() # done writing