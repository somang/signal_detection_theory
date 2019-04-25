import csv

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
all_list = []

spamReader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
uuids = {}
for row in spamReader:
    r = row[0].split(";")
    uuid, v_id, quiz, answer, caption = r[0], r[2], r[3], r[4], r[5]
    if not uuids.get(uuid):
        uuids[uuid] = {}
    if not uuids[uuid].get(caption):
        uuids[uuid][caption] = {}
    if not uuids[uuid][caption].get(quiz):
        uuids[uuid][caption][quiz] = {}
    uuids[uuid][caption][quiz] = answer

caption_vars = {}    
for i in uuids:
    user = uuids[i]
    tmp_row = [i] # for each row add uuid
    q1 = []
    q3 = []
    for v in user:        
        if not caption_vars.get(v):
            caption_vars[v] = []

        q1_ans = user[v]["Did you see any errors in the caption?"]
        q3_ans = user[v]["How would you rate the quality of the caption?"]
        # add answers for questions
        q1.append(to_score(q1_ans))
        q3.append(to_score(q3_ans))

    tmp_row += q1 + q3
    print(tmp_row)



# workbook = Workbook('q1q3.xlsx')
# worksheet = workbook.add_worksheet(i)    
for r in range(len(all_list)): # for each row
    row = all_list[r]
    print(row)
    #for c in range(len(row)): # for each column
    #    print(row[c])
#        worksheet.write(r,c,row[c]) # row, col, message
#workbook.close()