from data_collection import create_gt
from os import path
import json
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import gensim


def load_word2vec():
    return gensim.models.KeyedVectors.load_word2vec_format(r'cached_data/GoogleNews-vectors-negative300.bin',
                                                           binary=True)
def load_bert():
    # load pre-trained model (weights)
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert_model.eval()
    # load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return bert_model, tokenizer

def predict_mask(text, tokenizer, model):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_index = tokenized_text.index('[MASK]')

    predicted_indexes = torch.topk(predictions[0, masked_index], k=5)
    predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index.item()])[0] for predicted_index in
                       predicted_indexes.indices]
    return predicted_token


def similarity(values, compare_to, w2v_model):
    """
    compares the values to compare_to and finds the most appropriate answer
    :param values: (list of str) possible answers
    :param compare_to: (list of str) strings to compare to
    :param w2v_model: (gensim Word2Vec) a word2vec model
    :return: index of the correct answer
    """
    comps = [w2v_model.similarity(x, y)
             if y in w2v_model.vocab and x in w2v_model.vocab else 0
             for x in compare_to for y in values]
    return np.argmax(comps) % 4  # finds the best value


if __name__ == '__main__':
    if not path.exists(r'data_collection/psy_questions.json'):  # todo: add main config option to always overwrite
        create_gt()
    with open(r'data_collection/psy_questions.json', 'r') as f:
        gt = json.load(f)  # load gt
    # load word2vec model
    w2v_model = load_word2vec()
    # load BERT model
    bert_model, tokenizer = load_bert()

    total = len(gt)
    correct = 0

    pbar = tqdm(total=total)
    for example in gt:
        text = example['question']
        text = '[CLS] ' + text.replace('<MISSING>', '[MASK]') + ' [SEP]'  # for Bert
        words = predict_mask(text, tokenizer, bert_model)
        only_first_word = [x.split(' ')[0] for x in example['answers']]
        answer = str(similarity(only_first_word, words, w2v_model) + 1)
        if answer == example['real']:
            correct += 1
        pbar.update(1)
    print(f"""Out of {total} questions, {correct} were correct. So a {correct / total} accuracy score!""")
    pbar.close()
