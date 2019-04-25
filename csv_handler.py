import csv

class uqa():
    def __init__(self, uuid):
        self.uuid = uuid
    
    def __str__(self):
        return self.uuid + ""

    


filename = "q1q3_results.csv"
spamReader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
all_list = []
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
    for v in user:        
        if not caption_vars.get(v):
            caption_vars[v] = []
        
        for q in user[v]:
            print(q, user[v][q])





# workbook = Workbook('q1q3.xlsx')
# for i in hearing_groups:
#     worksheet = workbook.add_worksheet(i)    
#     for r in range(len(all_list)): # for each row
#         row = all_list[r]
#         for c in range(len(row)): # for each column
#             worksheet.write(r,c,row[c]) # row, col, message
# workbook.close()