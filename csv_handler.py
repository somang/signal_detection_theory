import csv
from xlsxwriter.workbook import Workbook
import xlrd
from xlrd import open_workbook
import pandas
from scipy.stats import shapiro


from scipy import stats

def to_score(answer):
    if answer == "1. Yes":
        return 1
    elif answer == "2. No":
        return 0
    elif answer == "1. Strongly dissatisfied":
        return 1
    elif answer == "2. Dissatisfied":
        return 2
    elif answer == "3. Neither satisfied nor dissatisfied":
        return 3
    elif answer == "4. Satisfied":
        return 4
    elif answer == "5. Strongly satisfied":
        return 5

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

for sn in xl.sheet_names:
    print(sn)

    df = pandas.read_excel(filename, sheet_name=sn, header=None, index_col=False)

    variations = {
        "Did you see any errors in the caption?": {},
        "How would you rate the quality of the caption?": {}
    }

    for i, r in df.iterrows():
        uuid, v_id, quiz, answer, caption = r[0], r[2], r[4], r[5], r[6]
        caption = caption.split("/")[1].split(".vtt")[0].split("_")
        video, caption_var = "_".join(caption[:2]), caption[2]
        # lets group by variations.
        if not variations[quiz].get(caption_var):
            variations[quiz][caption_var] = {}
        if not variations[quiz][caption_var].get(uuid):
            variations[quiz][caption_var][uuid] = answer

    # lets calculate t-tests over the columns we want,
    q1 = variations["Did you see any errors in the caption?"]

    ttest_result = list(map(lambda x: get_t_map(q1, 1, x), range(2,24))) # ttest( v1 , (v2-v23) )
    shapiro_result = list(map(lambda x: get_shapiro_map(q1, x), range(1, 24))) # check for normality
    
    # for r in ttest_result:
    #     print(str(r[1][1])) # variation vs t_result

    # for s in shapiro_result:
    #     print(str(s[1][1]))






































    # now we have a sorted table.
    all_list, uids = [], []
    for uid in variations["Did you see any errors in the caption?"]['1']: # get all uids
        uids.append(uid)

    for q in variations: # for each question
        qz = []
        head_row = ["uuid"]
        first = True
        for u in uids: # for each user
            row = [u]
            for n in range(1, 24): # from variation 1 to 23            
                var_n = str(n)
                if first:
                    head_row.append(var_n)
                ans = 100
                if n < 23: # because v23 was not added for some user
                    ans = variations[q][var_n][u] # add the answer of this user at this variation.
                elif n == 23:
                    if variations[q][var_n].get(u):
                        ans = variations[q][var_n][u]
                ans = to_score(ans)
                row.append(ans)
            if first:
                qz.append(head_row)
                first = False
            
            qz.append(row) # for each row, it will have a username, along with the answers for each variation number 1-23
        all_list.append(qz)# add the row to the queue

    ws_name = 'q1_' + sn
    worksheet = workbook.add_worksheet(ws_name)
    for r in range(len(all_list[0])): # for each row
        row = all_list[0][r]
        for c in range(len(row)): # for each column
            worksheet.write(r,c,row[c]) # row, col, message
    
    ws_name = 'q5_' + sn
    worksheet = workbook.add_worksheet(ws_name)
    for r in range(len(all_list[1])): # for each row
        row = all_list[1][r]
        for c in range(len(row)): # for each column
            worksheet.write(r,c,row[c]) # row, col, message

workbook.close() # done writing