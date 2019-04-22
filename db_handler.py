import sqlite3
from xlsxwriter.workbook import Workbook


def connect(sqlite_file):
    """ Make connection to an SQLite database file """
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def process_db(acceptable_id):
    n_userids = list(map(lambda x: x.strip(), acceptable_id.split(',')))    

    user_filter = "(r.interview_uuid = "
    for u in range(len(n_userids)):
        if u < len(n_userids)-1:
            user_filter += '"' + n_userids[u] + '" OR r.interview_uuid = '
        else:
            user_filter += '"' + n_userids[u] + '")'

    # now we connect database to extract some data into excel file.
    sqlite_file = 'db.sqlite3'
    conn, cur = connect(sqlite_file)
    
    #####################################################
    ### Now, let's see what I can do with the data.
    query = query = "SELECT DISTINCT r.interview_uuid, b.question_id, b.category, r.created, q.text, a.body, a.caption_var " +\
    "FROM ccvsite_response AS r, ccvsite_answerbase AS b, ccvsite_answerradio AS a, ccvsite_question AS q " +\
    "ON (r.id = b.response_id AND a.answerbase_ptr_id = b.id AND b.question_id = q.id)" +\
    " WHERE " + user_filter + ""    
    cur.execute(query)
    table = cur.fetchall()

    ansset_1 = ['1. Yes', '2. No']
    ansset_2 = ['1. Strongly disagree', '2. Disagree', '3. Neither agree nor disagree', '4. Agree', '5. Strongly agree']
    conditions = ['2. I identify as Deaf', '3. I am Deafened','4. I am Hard of Hearing']

    caption_set = {}
    for i in table:
        date, uuid, question_number, video, question, answer, caption = i[0],i[1],i[2],i[3],i[4],i[5],i[6]        
        if 'communities?' in question: # What statement best describes your relationship to the Deaf and/or Hard of Hearing communities?
            question = 'group'
        elif 'errors' in question: # Did you see any errors in the caption?
            question = 'error detection'
        elif 'confident' in question: # I am confident that my decision is correct:
            question = 'confidence level'
        elif 'quality' in question: # How would you rate the quality of the caption?
            question = 'caption quality'
        elif 'pleasure' in question: # How would you rate your viewing pleasure from the video?
            question = 'viewing pleasure'

        if answer in ansset_1:
            answer = answer.split(' ')[1]
            prev_answer = answer # save detectibility first.

        if caption == "demo":
            pass
        elif caption == None:
            pass
        else:
            caption = caption.split("/")[1].split(".")[0].split("_") #captions/v11_social_22.vtt
            content = caption[1]
            caption_var = caption[2]

        if caption_var not in caption_set: # if caption variation not in the set, then initialize
            caption_set[caption_var] = {}
            caption_set[caption_var]['group'] = {}
            caption_set[caption_var]['group']['2. I identify as Deaf'] = 0
            caption_set[caption_var]['group']['3. I am Deafened'] = 0
            caption_set[caption_var]['group']['4. I am Hard of Hearing'] = 0

            caption_set[caption_var]['error detection'] = {}
            caption_set[caption_var]['confidence level'] = {}
            caption_set[caption_var]['confidence level']['Yes'] = {}
            caption_set[caption_var]['confidence level']['No'] = {}

            caption_set[caption_var]['caption quality'] = {}
            caption_set[caption_var]['viewing pleasure'] = {}

            # q1
            caption_set[caption_var]['error detection']['Yes'] = 0
            caption_set[caption_var]['error detection']['No'] = 0
            # q2
            caption_set[caption_var]['confidence level']['Yes']['1. Strongly disagree'] = 0
            caption_set[caption_var]['confidence level']['Yes']['2. Disagree'] = 0
            caption_set[caption_var]['confidence level']['Yes']['3. Neither agree nor disagree'] = 0
            caption_set[caption_var]['confidence level']['Yes']['4. Agree'] = 0
            caption_set[caption_var]['confidence level']['Yes']['5. Strongly agree'] = 0

            caption_set[caption_var]['confidence level']['No']['1. Strongly disagree'] = 0
            caption_set[caption_var]['confidence level']['No']['2. Disagree'] = 0
            caption_set[caption_var]['confidence level']['No']['3. Neither agree nor disagree'] = 0
            caption_set[caption_var]['confidence level']['No']['4. Agree'] = 0
            caption_set[caption_var]['confidence level']['No']['5. Strongly agree'] = 0
            # q3
            caption_set[caption_var]['caption quality']['1. Strongly dissatisfied'] = 0
            caption_set[caption_var]['caption quality']['2. Dissatisfied'] = 0
            caption_set[caption_var]['caption quality']['3. Neither satisfied nor dissatisfied'] = 0
            caption_set[caption_var]['caption quality']['4. Satisfied'] = 0
            caption_set[caption_var]['caption quality']['5. Strongly satisfied'] = 0            
            # q4
            caption_set[caption_var]['viewing pleasure']['1. Very poor'] = 0
            caption_set[caption_var]['viewing pleasure']['2. Poor'] = 0
            caption_set[caption_var]['viewing pleasure']['3. Neither good nor poor'] = 0
            caption_set[caption_var]['viewing pleasure']['4. Good'] = 0
            caption_set[caption_var]['viewing pleasure']['5. Excellent'] = 0

        # add count.        
        if caption != "demo" and caption != None:
            if answer in ansset_2:
                caption_set[caption_var][question][prev_answer][answer] += 1     
            elif answer in conditions:
                caption_set[caption_var][question][answer] += 1                       
            else:
                caption_set[caption_var][question][answer] += 1
        
    sum_list = []
    for i, v in sorted(caption_set.items(), key=lambda x: int(x[0])):
        print("Variation #" + i)
        
        print('Yes', caption_set[i]['error detection']['Yes'])
        conf_label, conf_lvl, yn_label = [], [], []
        for l in caption_set[i]['confidence level']['Yes']:
            yn_label.append('Yes error detected in the CC.')
            conf_label.append(l)
            conf_lvl.append(caption_set[i]['confidence level']['Yes'][l])

        print('No', caption_set[i]['error detection']['No'])        
        conf_label_no, conf_lvl_no, yn_label_no = [], [], []
        for l in caption_set[i]['confidence level']['No']:
            yn_label_no.append('No error in the CC.')
            conf_label_no.append(l)
            conf_lvl_no.append(caption_set[i]['confidence level']['No'][l])

        conf_lvl_no.reverse()
        conf_label_no.reverse()
        yn_label_no.reverse()

        conf_lvl = conf_lvl + conf_lvl_no
        conf_label = conf_label + conf_label_no
        yn_label = yn_label + yn_label_no
        
        if i == '1': # first round,
            sum_list.append(yn_label)
            sum_list.append(conf_label)            
        sum_list.append(conf_lvl)
        
        print(conf_label)
        print(conf_lvl)
        print('---------------------------------------------------')
    
    return sum_list
        


if __name__ == "__main__":
    
    # filter out acceptable data only. to get rid of bad data.
    hoh = "981c026156904ab09124ecf24f22f308, ef7d53853ffa4de1a0ce9d5ccc62d90e, 8842de8026bb41dfbbda4c4449533432,\
        4f365caeba754ce2b64c05fe8bbf1b83, a0daa2a0a5334b1eb0167d437ee17cab, 2624763b44f74dee9296c724a89da38e,\
        213eb57e4d79488e97473d8269e21eaa, a5d0904aa4314081b4ebdf162f5e1ca0, 6516592bdd1c419e9510415b36121bea,\
        0c652f0835d04677adc6164fea034be2, 1b5e6927654a41bfacb38df795f65661, 7da0e901d854419da76b97e11ddbb1f5,\
        d45dad5247f4454aa317a3e314253fc4, 90bfe0ce60524ebd8fb127aa36f51b97, d1a31b5badc24250b424541b1441e524"
    
    deafened = "66e402d9d26542a39eb5242b76f96740, 1f3cf84fcb234574a0a0b3b55c9a6002, cbceee500b0844fc8665c59fb8268b87,\
        38dc1612584c4e33a2abd62451924d8c, 64c99d37023e4b25993a8b3a2b849bb1, 9f0bc4b929e44db9a5d36773978276a1,\
        1ccd0d04e4c24a9d9a28fcd7bfe5cb83, 20807d3f6bdc47cf9a59a4c986aea10a, e168d926f2a640aebb9048da56ed812e"
    
    deaf = "829843c8d16a4276b763dbd42d3529b7, 8ff52b0e75d444c6ae2ea36fa705b11e, 02b36ada868644588194b1686896d0a9,\
        9de393c400c4499f98717997361f2fed, eecfe1dae11b4cf2b09fe6c1e54774b5, c682e11225244671b7ba1e21852b2da0,\
        4f75600ef70c4f58803c5d4a4e3285b9, 35c2446de1b6479584470bff75dc45ee, 14ac5c3930164d67839097d8f64449af,\
        b7339943af1c4ee2b3bba8ba5e08c05b, fe412c5ac1544fd6aa627bdcf06b7982, ff0696e0e2bc4755af2215a733a5e292,\
        d51835f3aaa14e2b81a9f5c4fbacdaa2, 836ce291e0424e24bf38d60d0e2f5acf, 8ad718b2f12a4a7eaae9d26940ea4cfe,\
        8c3a587490834d43a6a6ddbfdf516a69, 6de765e89afe4ea6912ff3745b72d4a4, c45917462fad443fb0555ee51b5d3727,\
        60461644d8714d728dbbac307516c6ad, c6197608a4f1477ca6c6c0c17a3b976d, f3727975ca684e549b059aec62634e0b"
    
    # suspicious = "1f3cf84fcb234574a0a0b3b55c9a6002, cbceee500b0844fc8665c59fb8268b87, 38dc1612584c4e33a2abd62451924d8c,\
    #     1ccd0d04e4c24a9d9a28fcd7bfe5cb83, 9de393c400c4499f98717997361f2fed, c682e11225244671b7ba1e21852b2da0,\
    #     20807d3f6bdc47cf9a59a4c986aea10a, ff0696e0e2bc4755af2215a733a5e292"

    # mturk_deaf = "829843c8d16a4276b763dbd42d3529b7, 8ff52b0e75d444c6ae2ea36fa705b11e, 02b36ada868644588194b1686896d0a9,\
    #     9de393c400c4499f98717997361f2fed, eecfe1dae11b4cf2b09fe6c1e54774b5, c682e11225244671b7ba1e21852b2da0,\
    #     4f75600ef70c4f58803c5d4a4e3285b9, fe412c5ac1544fd6aa627bdcf06b7982,ff0696e0e2bc4755af2215a733a5e292"

    # deaf = "35c2446de1b6479584470bff75dc45ee, 14ac5c3930164d67839097d8f64449af,\
    #     b7339943af1c4ee2b3bba8ba5e08c05b,\
    #     d51835f3aaa14e2b81a9f5c4fbacdaa2, 836ce291e0424e24bf38d60d0e2f5acf, 8ad718b2f12a4a7eaae9d26940ea4cfe,\
    #     8c3a587490834d43a6a6ddbfdf516a69, 6de765e89afe4ea6912ff3745b72d4a4, c45917462fad443fb0555ee51b5d3727,\
    #     60461644d8714d728dbbac307516c6ad, c6197608a4f1477ca6c6c0c17a3b976d, f3727975ca684e549b059aec62634e0b"

    # hoh = "981c026156904ab09124ecf24f22f308, ef7d53853ffa4de1a0ce9d5ccc62d90e, 8842de8026bb41dfbbda4c4449533432,\
    #     4f365caeba754ce2b64c05fe8bbf1b83, a0daa2a0a5334b1eb0167d437ee17cab, 2624763b44f74dee9296c724a89da38e,\
    #     213eb57e4d79488e97473d8269e21eaa, a5d0904aa4314081b4ebdf162f5e1ca0, 6516592bdd1c419e9510415b36121bea,\
    #     0c652f0835d04677adc6164fea034be2, 1b5e6927654a41bfacb38df795f65661, 7da0e901d854419da76b97e11ddbb1f5,\
    #     d45dad5247f4454aa317a3e314253fc4, 90bfe0ce60524ebd8fb127aa36f51b97, d1a31b5badc24250b424541b1441e524"
    
    # deafened = "66e402d9d26542a39eb5242b76f96740, 64c99d37023e4b25993a8b3a2b849bb1, 9f0bc4b929e44db9a5d36773978276a1, e168d926f2a640aebb9048da56ed812e"
    

    hearing_groups = {
        "all": deafened + ", " + hoh + ", " + deaf,
        "deaf": deaf,
        "hoh_deafened": deafened + ", " + hoh
    }

    workbook = Workbook('output.xlsx')
    for i in hearing_groups:
        worksheet = workbook.add_worksheet(i)
        all_list = process_db(hearing_groups[i])

        for r in range(len(all_list)): # for each row
            row = all_list[r]
            for c in range(len(row)): # for each column
                worksheet.write(r,c,row[c]) # row, col, message
    
    workbook.close()
    print("done exporting.")
