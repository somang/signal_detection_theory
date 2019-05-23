SELECT DISTINCT r.interview_uuid, b.question_id, b.category, r.created, q.text, a.body, a.caption_var
FROM ccvsite_response AS r, ccvsite_answerbase AS b, ccvsite_answerradio AS a, ccvsite_question AS q
ON (r.id = b.response_id AND a.answerbase_ptr_id = b.id AND b.question_id = q.id)