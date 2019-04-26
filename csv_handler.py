import csv
from xlsxwriter.workbook import Workbook


class uqa():
    def __init__(self, uuid):
        self.uuid = uuid
    
    def __str__(self):
        return self.uuid + ""

    

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
    

filename = "q1q3_results.csv"

spamReader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')

variations = {
        "Did you see any errors in the caption?": {},
        "How would you rate the quality of the caption?": {}
    }

# lets group by variations.
for row in spamReader:
    r = row[0].split(";")
    uuid, v_id, quiz, answer, caption = r[0], r[2], r[3], r[4], r[5]
    caption = caption.split("/")[1].split(".vtt")[0].split("_")
    video, caption_var = "_".join(caption[:2]), caption[2]

    if not variations[quiz].get(caption_var):
        variations[quiz][caption_var] = {}
    if not variations[quiz][caption_var].get(uuid):
        variations[quiz][caption_var][uuid] = answer

# for q in variations: # for each question,
#     for v in range(1, len(variations[q]) + 1): # for each variation
        #variations[q][str(v)] = sorted(variations[q][str(v)])

# now we have a sorted table.
all_list = []
head_row = ["uuid"]
first = True
uids = []

for uid in variations["Did you see any errors in the caption?"]['1']: # get all uids
    uids.append(uid)

for q in variations: # for each question
    qz = []
    for u in uids: # for each user
        row = [u]
        for n in range(1, 24): # from variation 1 to 23            
            var_n = str(n)
            if first:
                head_row.append(var_n)
            ans = 100
            if n < 23:
                ans = variations[q][var_n][u]
            elif n == 23:
                if variations[q][var_n].get(u):
                    ans = variations[q][var_n][u]
            #ans = to_score(ans)
            row.append(ans)
        if first:
            qz.append(head_row)
            first = False
        
        qz.append(row)
    all_list.append(qz)


workbook = Workbook('q1q3.xlsx')
worksheet = workbook.add_worksheet('q1')
for r in range(len(all_list[0])): # for each row
    row = all_list[0][r]
    for c in range(len(row)): # for each column
        worksheet.write(r,c,row[c]) # row, col, message

worksheet = workbook.add_worksheet('q3')
for r in range(len(all_list[1])): # for each row
    row = all_list[1][r]
    for c in range(len(row)): # for each column
        worksheet.write(r,c,row[c]) # row, col, message
workbook.close()