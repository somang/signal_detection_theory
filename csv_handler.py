import csv

class uqa():
    def __init__(self, uuid):
        self.uuid = uuid
    
    def __str__(self):
        return self.uuid + ""

    


filename = "q1q3_results.csv"
spamReader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')

for row in spamReader:
    r = row[0].split(";")
    uuid, v_id, quiz, answer, caption = r[0], r[2], r[3], r[4], r[5]
    print(uuid, v_id, caption)
    print(quiz, answer)
