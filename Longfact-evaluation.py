# Itemnize the statement, then judge the trufulness.
import json
from nltk import tokenize
import re
import numpy as np

def detect_initials(text):
    pattern = r'[A-Z]\. ?[A-Z]\.'
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    """Fix sentence splitter issues."""
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split('.') if t.strip()]

            for i, (sent1, sent2) in enumerate(
                zip(curr_sentences, curr_sentences[1:])
            ):
                if sent1.endswith(alpha1 + '.') and sent2.startswith(alpha2 + '.'):
                    # merge sentence i and i+1
                    curr_sentences = (
                        curr_sentences[:i]
                        + [curr_sentences[i] + ' ' + curr_sentences[i + 1]]
                        + curr_sentences[i + 2 :]
                    )
                    break
    sentences, combine_with_previous = [], None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def get_sentences(generation):
    paragraphs = [para.strip() for para in generation.split('\n') if para.strip()]
    sentences, para_breaks = [], []

    for para_idx, paragraph in enumerate(paragraphs):
        if para_idx > 0:
            para_breaks.append(len(sentences))

        initials = detect_initials(paragraph)
        curr_sentences = tokenize.sent_tokenize(paragraph)
        curr_sentences_2 = tokenize.sent_tokenize(paragraph)
        curr_sentences = fix_sentence_splitter(curr_sentences, initials)
        curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)
        # ensure the crediability of the sentence splitter fixing algorithm
        assert curr_sentences == curr_sentences_2, (
            paragraph,
            curr_sentences,
            curr_sentences_2,
        )
        sentences += curr_sentences
    return sentences

from openai import OpenAI
OPENAI_API_KEY = "YOUR OPENAI KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

def main(filepath):
    with open(filepath,'r') as file:
        data = json.load(file)
    for i in range(len(data['model_completion'])):
        ans = data['model_completion'][i]
        if 'Q:' in ans:
            ans = ans.split('Q:')[0]
        sentences = get_sentences(ans)
        res,correct,incorrect=[],[],[]
        for sentence in sentences:
            prompt = f"""Instructions:\n1. You are given a sentence. Your task is to break the sentence down into a list of atomic facts.\n2. An atomic fact is a sentence containing a singular piece of information.\n3. Each atomic fact in the outputted list should check a different piece of information.\n4. Use the previous examples to learn how to do this.\n5. You should only output the atomic facts as a list, with each item starting with “- ”. Do not include other formatting.\n6. Your task is to do this for the last sentence that is given.\nPlease breakdown the following sentence into independent facts:\nExamples:\nDuring his professional career, McCoy played for the Broncos, the San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars.\n- McCoy played for the Broncos.\n- McCoy played for the Broncos during his professional career.\n- McCoy played for the San Diego Chargers.\n- McCoy played for the San Diego Chargers during his professional career.\n- McCoy played for the Minnesota Vikings.\n- McCoy played for the Minnesota Vikings during his professional career.\n- McCoy played for the Jacksonville Jaguars.\n- McCoy played for the Jacksonville Jaguars during his professional career.\n\nHe played college football for the University of Oregon, where he was an All-Pac-12 selection and was named to the All-America team in 2016.\n- He played college football.\n- He played college football for the University of Oregon.\n- He was an All-Pac-12 selection.\n- He was an All-Pac-12 selection at the University of Oregon.\n- He was named to the All-America team.\n- He was named to the All-America team in 2016.\n- He was named to the All-America team in 2016 at the University of Oregon.\n\nHis breakthrough came with the leading role in the acclaimed crime-drama film Memories of Murder in 2003.\nAtomic facts:\n- His breakthrough came with Memories of Murder.\n- He was the leading role in Memories of Murder.\n- Memories of Murder was released in 2003.\n- Memories of Murder is a film.\n- Memories of Murder is an acclaimed crime-drama film.\n\nPlease breakdown the following sentence into independent facts:\n{sentence}\nAtomic facts:"""
            completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
            sentences_list = completion.choices[0].message.content.splitlines()
            sentences_list = [sen.lstrip('- ').strip() for sen in sentences_list]
            res += sentences_list
        data['is_correct'].append({"no":i,"atom":res,"correct":correct,"incorrect":incorrect,"sentences":sentences})

    with open(filepath, "w") as file:
        json.dump(data, file)
    with open(filepath,'r') as file:
        data = json.load(file)
    for i in range(len(data['is_correct'])):
        correct,incorrect,hallu=[],[],[]
        for j in range(len(data['is_correct'][i]["atom"])):
            prompt = f"""{data['model_completion'][i]}\nRead the above text carefully. Note that some of the information in it might be incorrect.\nIn this text, is the claim "{data['is_correct'][i]["atom"][j]}" factual and correct?\nYour response should either "Yes" or "No"."""
            completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
            )
            ans = completion.choices[0].message.content
            if "Yes" in ans or "yes" in ans:
                correct.append(data['is_correct'][i]["atom"][j])
            elif "No" in ans or "no" in ans:
                incorrect.append(data['is_correct'][i]["atom"][j])
        data['is_correct'][i]["correct"]=correct
        data['is_correct'][i]["incorrect"]=incorrect
        if i%2==0:
            with open(filepath, "w") as file:
                json.dump(data, file)
    with open(filepath, "w") as file:
        json.dump(data, file)

file_paths=['FILE PATH']
for filepath in file_paths:
    main(filepath)