import csv
from xlsxwriter.workbook import Workbook
import xlrd
from xlrd import open_workbook
import pandas
from scipy.stats import shapiro


from scipy import stats

def export_excel(worksheet, all_list):
    for r in range(len(all_list)): # for each row
        row = all_list[r]
        for c in range(len(row)): # for each column
            worksheet.write(r,c,row[c]) # row, col, message
    


def to_score(answer):
    
    # calculate the cognitive score
    # if a person detects an error, it will be 'yes' otherwise 'no'
    # 'yes' = 1, and 'no' = -1
    # then, the five confidence level can be considered as weights
    #
    # Q2: I am confident that my decision is correct:
    # '1. Strongly disagree': 20%
    # '2. Disagree': 40%
    # '3. Neither agree nor disagree': 60%
    # '4. Agree': 80$
    # '5. Strongly agree': 100%
    # weights are multiplied to the score.
    # 
    if answer == "1. Strongly dissatisfied":
        return 1
    elif answer == "2. Dissatisfied":
        return 2
    elif answer == "3. Neither satisfied nor dissatisfied":
        return 3
    elif answer == "4. Satisfied":
        return 4
    elif answer == "5. Strongly satisfied":
        return 5
    
    if answer == "1. Strongly disagree":
        return 0.2
    elif answer == "2. Disagree":
        return 0.4
    elif answer == "3. Neither agree nor disagree":
        return 0.6
    elif answer == "4. Agree":
        return 0.8
    elif answer == "5. Strongly agree":
        return 1

    if answer == "1. Yes":
        return 1
    elif answer == "2. No":
        return -1

    
def get_weight_score(x):
    if x[1] != None:
        return float(x[0]) * float(x[1])
    

def get_t_map(question, v1, v2):
    l1 = list(map(lambda x: to_score(x), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x), question[str(v2)].values()))
    return (v2, stats.ttest_ind(l1, l2))

def get_t(question, v1, v2):
    l1 = list(map(lambda x: to_score(x), question[str(v1)].values()))
    l2 = list(map(lambda x: to_score(x), question[str(v2)].values()))
    return stats.ttest_ind(l1, l2)
    
def get_shapiro_map(question, v1):
    l1 = list(map(lambda x: to_score(x), question[str(v1)].values()))
    return (v1, shapiro(l1))

workbook = Workbook('q1q3_handled.xlsx') # output


filename = "q1q3_results.xlsx"
xl = pandas.ExcelFile(filename)

"""
Is narrator in the video clip?

v2_sports: no
v3_sports: no
v4_hockey: no
v5_hockey: no
v6_hockey: no
v7_weather: yes, but cannot see face
v8_weather: yes, but cannot see lips clearly
v9_weather: yes, but cannot see face


v1_sports: yes
v10_social: yes
v11_social: yes
v12_social: yes
v13_weather: yes
v14_weather: yes
v15_weather: yes
v16_weather: yes
v17_weather: yes
v18_weather: yes
v19_weather: yes
v20_weather: yes
v21_weather: yes
v22_weather: yes
v23_social: yes

"""


narrator_vid = ["v1", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"]
no_narrator_vid = ["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]


for sn in xl.sheet_names:
    print(sn)

    df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)

    variations = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {}
    }

    nar = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {}
    }

    no_nar = {
        "Did you see any errors in the caption?": {},
        "I am confident that my decision was correct:": {},
        "How would you rate the quality of the caption?": {}
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
                nar[quiz][caption_var][uuid] = answer            
        elif video in no_narrator_vid:
            if not no_nar[quiz].get(caption_var):
                no_nar[quiz][caption_var] = {}
            if not no_nar[quiz][caption_var].get(uuid):
                no_nar[quiz][caption_var][uuid] = answer

        # lets group by variations.
        if not variations[quiz].get(caption_var):
            variations[quiz][caption_var] = {}
        if not variations[quiz][caption_var].get(uuid):
            variations[quiz][caption_var][uuid] = answer


    ## lets calculate t-tests over the columns we want,
    #q1 = variations["Did you see any errors in the caption?"]
    #ttest_result = list(map(lambda x: get_t_map(q1, 1, x), range(2,24))) # ttest( v1 , (v2-v23) )
    #shapiro_result = list(map(lambda x: get_shapiro_map(q1, x), range(1, 24))) # check for normality
    # for r in ttest_result:
    #     print(str(r[1][1])) # variation vs t_result
    # for s in shapiro_result:
    #     print(str(s[1][1]))

    # now we have a sorted table.
    all_list, uids = [], []
    for uid in variations["Did you see any errors in the caption?"]['1']: # get all uids
        uids.append(uid)

    
    for q in variations:
        for u in uids:
            for n in range(1,24): # caption variations
                var_n = str(n)
                if nar[q][var_n].get(u):
                    print("nar", u, nar[q][var_n][u])
                elif no_nar[q][var_n].get(u):
                    print("no nar", u, no_nar[q][var_n][u])

















#     for q in variations: # for each question
#         qz = []
#         head_row = ["uuid"]
#         first = True
#         for u in uids: # for each user
#             row = [u]
#             for n in range(1, 24): # from variation 1 to 23            
#                 var_n = str(n)
#                 if first:
#                     head_row.append(var_n)
#                 ans = 100
#                 if n < 23: # because v23 was not added for some user
#                     ans = variations[q][var_n][u] # add the answer of this user at this variation.
#                 elif n == 23:
#                     if variations[q][var_n].get(u):
#                         ans = variations[q][var_n][u]
#                 ans = to_score(ans)
#                 row.append(ans)
#             if first:
#                 qz.append(head_row)
#                 first = False
#             qz.append(row) # for each row, it will have a username, along with the answers for each variation number 1-23
#         all_list.append(qz)# add the row to the queue


#     # now that we have the raw values, let's calculate the weighted scores.
#     q4 = []
#     for r1 in all_list[0]:
#         for r2 in all_list[1]:
#             if (r1[0] == r2[0]) and (r1[0] != "uuid"):
#                 new_val_row = r1[:1] + list(map(lambda x: get_weight_score(x), zip(r1[1:], r2[1:])))
#                 q4.append(new_val_row)
#     all_list.append(q4)            

#     worksheet = workbook.add_worksheet('q1_' + sn)
#     export_excel(worksheet, all_list[0])
#     worksheet = workbook.add_worksheet('q2_' + sn)
#     export_excel(worksheet, all_list[1])
#     worksheet = workbook.add_worksheet('q3_' + sn)
#     export_excel(worksheet, all_list[2])
#     worksheet = workbook.add_worksheet('q4_' + sn)
#     export_excel(worksheet, all_list[3])

# workbook.close() # done writing