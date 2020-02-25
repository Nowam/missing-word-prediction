from data_collection import crawl_psy
import re
from tika import parser
import os
import json


def create_gt(crawl=True):
    """
    extract the missing word questions from a typical psycometric test
    creates a two json files, one with test names (mainly for debugging) and one without
    :param crawl: whether to redownload the files
    :return: None
    """
    if crawl:
        crawl_psy()  # Get tests from da web
    directory = 'tests'
    data = []
    data_with_tests = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"extracting from: {filename}")
            test_text = extract_text_from_pdf(directory, filename)
            all_questions = extract_questions(test_text)
            for part in all_questions:
                data.extend(all_questions[part])
            data_with_tests[filename] = []
            data_with_tests[filename].append(all_questions)
    data = clean_data(data)
    save_json(data, r'psy_questions.json')
    save_json(data_with_tests, r'psy_questions_full.json')


def extract_questions(test):
    """
    parses and extracts questions from a psy-test in
    :param test: (str) all the text in the test
    :return: all the questions found in test, split into parts
    """
    ret = {}
    part = 1
    # This is just annoying man
    test = test.replace(
        """כל הזכויות שמורות למרכז ארצי לבחינות ולהערכה )ע"ר(  ©\nאין להעתיק או להפיץ בחינה זו או קטעים ממנה, בכל צורה ובכל אמצעי, או ללמדה, כולה או חלקים ממנה, בלא אישור בכתב מהמרכז הארצי לבחינות ולהערכה.\n\nמועד אפריל 2010 אנגלית - פרק ראשון""",
        "")
    real_answers = test.split('מפתח תשובות נכונות\n')[1].split('אנגלית')
    for chapter in test.split('ENGLISH')[1:]:  # each test has at least two parts of English chapters
        completion = chapter.split('Sentence Completions')[1].split('Questions ')[1]

        full_questions = re.findall(r'\n\n(\d{1,2})\. {1,2}([\s\S]*?\.)[\s]*?\n( ?)+\n',
                                    completion)  # Find full question string
        question_nums = [q[0] for q in full_questions]  # question numbers

        questions = [q[1].replace('\n', '') for q in full_questions]
        questions = [re.sub(r'(^ |  |(?=^[a-z])| (?=[\.,;]))', ' <MISSING> ', q)
                     for q in questions]  # Put missing where words are missing

        answers = re.findall(r'(\(1\)[\s\S]+?)\n ?\n[^\(]', completion)
        answers = [[s[0] for s in re.findall(r'\(\d\)[ \t]+(\w+( \w+)?)', ans)] for ans in
                   answers]  # Extract each possible answer

        parsed_answers = {str(x + 1): v for x, v in enumerate(re.findall(r'(\d+?)הנכונה', real_answers[part])[0])}

        ret[f'part_{part}'] = [{"question_num": q_num, "question": q, "answers": ans, "real": parsed_answers[q_num]} for
                               q_num, q, ans in zip(question_nums, questions, answers)]
        part += 1
    return ret


def clean_data(data):
    """
    cleans data from problematic examples, and from question numbers
    :param data:
    :return:
    """
    for idx, example in enumerate(data):
        if len(example['answers']) != 4:
            del data[idx]
        if 'MISSING' not in example['question']:
            del data[idx]
        del example['question_num']
    return data


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def extract_text_from_pdf(directory, filename):
    raw = parser.from_file(os.path.join(directory, filename))
    return raw['content']


if __name__ == '__main__':
    create_gt(crawl=False)
