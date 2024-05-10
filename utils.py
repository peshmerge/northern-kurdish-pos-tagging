import pickle
import zipfile
import os
import random

# Must be done once!
# import nltk
# nltk.download('punkt')

from flair.data import Sentence
from klpt.tokenize import Tokenize
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer

vectoriser = DictVectorizer(sparse=False)
REPLACEMENT_SYMBOL = "None"
PRECISION = 2
random.seed(42)


def load_data_from_pickle_file(input_path):
    with open(input_path, "rb") as file:
        return pickle.load(file)


def predict_pos_tags_using_flair_model(model, tokens_list):
    """ Function to predict POS tags using a model trained with Flair """
    flair_sentence = Sentence(tokens_list)
    model.predict(flair_sentence)
    output = flair_sentence.to_dict()
    list_tokens_tags = []
    for item in output['tokens']:
        token = item['text']
        if len(item['labels']):
            tag = item['labels'][0]['value']
        else:
            tag = 'None'
        list_tokens_tags.append((token, tag))
    return list_tokens_tags


def unzip_extra_trees_pos_model(model_name, input_path, output_path):
    """ Unzip zipped extraTrees file"""
    if not os.path.exists(model_name + ".pickle"):
        with zipfile.ZipFile(input_path, "r") as zip_file:
            zip_file.extractall(path=output_path)
    else:
        print(f"{model_name}.pickle already exists, therefore not decompressing!")


def tokenize_sentence(sentence, tokenization_method):
    """ Tokenize a sentence using either rKLPT or NLTK word_tokenize """
    tokens = []
    if tokenization_method == "KLPT":
        tokenizer = Tokenize("Kurmanji", "Latin")
        tokens = tokenizer.word_tokenize(sentence, separator=' ', mwe_separator=' ', keep_form=True)
    elif tokenization_method == "NLTK":
        tokens = word_tokenize(sentence)
    return tokens


def load_extra_trees_vectoriser(path, training_data_type):
    """ Load extraTrees vectoriser"""
    return load_data_from_pickle_file(f"{path}/KMR_POS_ExtraTrees_Vectoriser_{training_data_type}.pickle")


def extract_features(sentences):
    """ This function expects a list of sentences where each sentence is a list of tuples (token,tag) """
    features = []
    for sentence in sentences:

        for index in range(len(sentence)):
            k = index
            features_dict = {}
            token, tag = sentence[k]
            is_number = False
            try:
                if float(token):
                    is_number = True
            except:
                pass

            features_dict['token'] = token
            features_dict['tag'] = tag
            features_dict['lower_cased_token'] = token.lower()
            features_dict['suffix1'] = token[-1]
            features_dict['suffix2'] = token[-2:]
            features_dict['suffix3'] = token[-3:]
            features_dict['is_capitalized'] = token.isalpha() and token[0].isupper()
            features_dict['is_number'] = is_number
            features_dict['is_first'] = k == 0
            features_dict['is_last'] = k == len(sentence) - 1

            # print(f"K value outside {k}")
            if k == 0:
                # print(f" K==0 {sentence[k]} {sentence[k-1]}")
                features_dict['prev_tag'] = "<start_tag>"
                features_dict['prev_prev_tag'] = "<start_tag>_<start_tag>"
                features_dict['prev_token'] = "<start_token>"

            if k == 1:
                # print(f" K==1 {sentence[k]} {sentence[k-1]}")
                token1, tag1 = sentence[k - 1]
                features_dict['prev_tag'] = tag1
                features_dict['prev_prev_tag'] = "<start_tag>_" + tag1
                features_dict['prev_token'] = token1
            elif k > 1:
                # print(f" K>1 {sentence[k-1]} {sentence[k-2]}")
                token1, tag1 = sentence[k - 1]
                token2, tag2 = sentence[k - 2]
                features_dict['prev_tag'] = tag1
                features_dict['prev_prev_tag'] = tag2 + "_" + tag1
                features_dict['prev_token'] = token1

            if k < len(sentence) - 1:
                token, tag = sentence[k + 1]
                features_dict['next_tag'] = tag
                features_dict['next_token'] = token
            features.append(features_dict)
    return features


# expects [(token,tag),(token,tag)]
def extract_tokens(token_tags_lists):
    return [item[0] for item in token_tags_lists]


# expects [(token,tag),(token,tag)]
def extract_tags(token_tags_lists):
    return [item[1] for item in token_tags_lists]
