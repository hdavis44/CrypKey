import pandas as pd
import numpy as np
from termcolor import colored
import time

start_time = time.time()
import spacy
time_diff = round((time.time() - start_time), 4)
print(colored(f'* Importing SPACY took {time_diff} sec.', 'blue'))

from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.data import get_local_data


# https://spacy.io/
# pip install spacy==3.2.0
# python -m spacy download en_core_web_sm


def spacy_function(txt):
    result = ""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(txt)

    # characters
    tot_chars = sum([len(token.lemma_) for token in doc])
    tot_chars = 1 if tot_chars == 0 else tot_chars
    # aplpha characters
    tot_alpha_chars = sum(
        [len(token.lemma_) for token in doc if token.is_alpha == True])
    tot_alpha_chars = 1 if tot_alpha_chars == 0 else tot_alpha_chars
    # words
    tot_words = len(doc)
    tot_words = 1 if tot_words == 0 else tot_words
    # alpha words
    tot_aplpha_words = len(
        [token.lemma_ for token in doc if token.is_alpha == True])
    tot_aplpha_words = 1 if tot_aplpha_words == 0 else tot_aplpha_words
    # unique alpha words
    unique_alpha_words = len(
        set([token.lemma_ for token in doc if token.is_alpha == True]))
    # stop words
    tot_stop_words = len(
        [token.lemma_ for token in doc if token.is_stop == True])
    tot_stop_words = 1 if tot_stop_words == 0 else tot_stop_words
    # unique stop words
    unique_stop_words = len(
        set([token.lemma_ for token in doc if token.is_stop == True]))
    # verbs
    tot_verbs = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    tot_verbs = 1 if tot_verbs == 0 else tot_verbs
    # unique verbs
    unique_verbs = len(
        set([token.lemma_ for token in doc if token.pos_ == "VERB"]))
    # adjectives
    tot_adj = len([token.lemma_ for token in doc if token.pos_ == "ADJ"])
    tot_adj = 1 if tot_adj == 0 else tot_adj
    # unique adjectives
    unique_adj = len(
        set([(token.lemma_) for token in doc if token.pos_ == "ADJ"]))
    # nouns
    tot_nouns = len([token.lemma_ for token in doc if token.pos_ == "NOUN"])
    tot_nouns = 1 if tot_nouns == 0 else tot_nouns
    # unique nouns
    unique_nouns = len(
        set([token.lemma_ for token in doc if token.pos_ == "NOUN"]))
    # symbols
    tot_symbols = len([token.lemma_ for token in doc if token.pos_ == "SYM"])
    # numbers
    tot_numbers = len([token.lemma_ for token in doc if token.pos_ == "NUM"])
    # sentences
    tot_sents = len([token for token in doc.sents])
    tot_sents = 1 if tot_sents == 0 else tot_sents

    ## ENGINEERED FEATURES
    sentence_len = tot_sents / tot_words
    word_len = tot_aplpha_words / tot_alpha_chars
    nouns = tot_nouns / tot_aplpha_words
    noun_richness = unique_nouns / tot_nouns
    verbs = tot_verbs / tot_aplpha_words
    verb_richness = unique_verbs / tot_verbs
    adjs = tot_adj / tot_aplpha_words
    adj_richness = unique_adj / tot_adj
    stopws = tot_stop_words / tot_aplpha_words
    stopw_richness = unique_stop_words / tot_stop_words
    vocab_richness = unique_alpha_words / tot_aplpha_words
    nums = tot_numbers / tot_words
    symbs = tot_symbols / tot_words
    alpha_chrs = tot_alpha_chars / tot_chars

    ## ASSEMBLE RESULT
    result_list = [
        sentence_len, word_len, nouns, noun_richness, verbs, verb_richness,
        adjs, adj_richness, stopws, stopw_richness, vocab_richness, nums,
        symbs, alpha_chrs
    ]

    result = ""
    for i in result_list:
        result += str(i) + " "

    return result[:-1]


def get_engineered_df(raw_text:pd.Series):
    headers = [
        'sentence_len', 'word_len', 'nouns', 'noun_richness', 'verbs',
        'verb_richness', 'adjs', 'adj_richness', 'stopws', 'stopw_richness',
        'vocab_richness', 'nums', 'symbs', 'alpha_chrs'
    ]
    start_time = time.time()

    engineered = raw_text.apply(spacy_function)
    df_engineered = engineered.str.split(" ", expand=True,)
    df_engineered.columns = headers

    time_diff = round((time.time() - start_time), 4)
    print(colored(f'* running the SPACY feat.eng. took {time_diff} sec.', 'blue'))

    return df_engineered


def get_engineered_df_from_text(text):
    return get_engineered_df(pd.Series(text))


def _pre_proc_engineering(text):
    '''
    this function is to be used by get_extended_eng_features_df()
    '''
    text = text.replace('\n', ' ')
    word_list = text.split()
    no_of_words = len(word_list)
    no_of_words = 1 if no_of_words == 0 else no_of_words
    result = ''

    for char in "@#!?":
        result += str(text.count(char) / no_of_words) + " "

    capitalized_words = 0
    upper_words = 0

    for word in word_list:
        if word.capitalize() == word:
            capitalized_words += 1
        if word.upper() == word:
            upper_words += 1
    result += str(capitalized_words / no_of_words) + ' ' + str(
        upper_words / no_of_words)

    return result


def _post_proc_engineering(text):
    '''
    this function is to be used by get_extended_eng_features_df()
    '''
    # frequency of two most common words
    word_list = text.split()
    if len(word_list) == 0:
        return 0

    word_dict = {}
    for word in word_list:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

    word_dict_list = sorted(word_dict, key=word_dict.get, reverse=True)

    if len(word_dict_list) == 1:
        return word_dict[word_dict_list[0]] / len(word_list)

    return (word_dict[word_dict_list[0]] +
            word_dict[word_dict_list[1]]) / len(word_list)


def get_extended_eng_features_df(raw_text: pd.Series, preproc_text=None):
    # Feature extraction from raw text
    engineered = raw_text.apply(_pre_proc_engineering)
    engineered_df = engineered.str.split(" ", expand=True)
    engineered_df.columns = ['amt_@', 'amt_#', 'amt_!',
                             'amt_?', 'amt_capitalized', 'amt_upper']
    engineered_df = engineered_df.astype(float)

    # Run preprocessing on text - or use from imput
    if isinstance(preproc_text, pd.Series):
        preprocessed_df = preproc_text
    else:
        preprocessed_df = TextPreprocessor().transform(raw_text)

    # Feature extraction on preprocessed text
    engineered_df['pop_word_freq'] = preprocessed_df.apply(
        _post_proc_engineering)

    # Create or Load Spacy engineering features
    try:
        X_spacy = get_local_data(data_file_name='us_election_eng.csv___')
        X_spacy = X_spacy.iloc[:len(raw_text), 1:]
        #X_spacy.drop(columns=['Unnamed: 0'], inplace=True)
    except (IOError, AttributeError) as e:
        X_spacy = get_engineered_df(raw_text)
        print(colored('- New data has been created', 'green'))


    # Concatinate all features
    final_eng_df = pd.concat([X_spacy, engineered_df], axis=1)

    # return dataframe
    return final_eng_df


if __name__ == '__main__':
    from detecting_fake_news.data import get_local_data
    df = get_local_data(data_file_name='us_election.csv', nrows=110)

    eng_df = get_engineered_df(df['text'])

    print(eng_df.head(3))
