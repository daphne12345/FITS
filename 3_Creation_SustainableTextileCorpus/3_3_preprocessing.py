import fitz
import pandas as pd
import os
import re
import spacy
from nltk.tokenize import sent_tokenize
from langdetect import detect
import nltk
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def extract_text(filename):
    """
    Extracts the text from the input file.
    :param filename: path to the file
    :return: the extracted raw text
    """
    try:
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        return None
    
def read_pdfs(path):
    """
    Reads all pdf files within the given folder to a dataframe.
    :param path: path to the files
    :return: dataframe of raw texts 
    """
    df = pd.DataFrame()
    df['filename'] = os.listdir(path)
    df['filename'] = path + df['filename']
    df['text'] = df['filename'].apply(extract_text)
    df = df[df['text'].notna()]

    if df.shape[0] < 1:
        raise Exception('Empty Dataframe')
    return df

def drop_duplicates(df):
    """
    Drops duplicates by lower case text.
    :param df: dataframe
    :return: dataframe without duplciates 
    """
    df['text_low'] = df['text'].apply(lambda text: ''.join([t.lower() for t in text]))
    df = df.drop_duplicates(subset='text_low')  # drop articles with the same abstract
    df = df.drop(columns=['text_low'])
    return df

def remove_special_characters(df):
    """
    Removes special characters, double spaces and linebreaks.
    :param df: dataframe
    :return: dataframe without speical characters 
    """
    df['text'] = df['text'].apply(lambda text: text.replace('\n', ''))
    df['text'] = df['text'].apply(lambda text: re.sub(' +', ' ', str(text)))  # replace multiple spaces
    df['text'] = df['text'].apply(lambda text: re.sub('\. +', '.', str(text)))  # replace multiple periods
    df['text'] = df['text'].apply(lambda text: re.sub('\.+', '. ', str(text)))  # replace multiple periods
    return df

def remove_copyright(df):
    """
    Sorts out everything after (c) or ©.
    :param df: dataframe
    :return: dataframe without copyright infos 
    """
    df['text'] = df['text'].apply(lambda text: ''.join(text.split('©')[:-1]) if '©' in text else text)
    remove = df['text'].apply(lambda text: '(c)' + text.split('(c)')[-1] if '(c)' in text else text).value_counts()
    remove = list(remove[remove > 1].index)
    remove2 = df['text'].apply(lambda text: '(C)' + text.split('(C)')[-1] if '(C)' in text else text).value_counts()
    remove = remove + list(remove2[remove2 > 1].index)
    regex = re.compile('|'.join(map(re.escape, remove)))
    df['text'] = df['text'].apply(lambda text: regex.sub('', text))
    return df

def drop_short_texts(df):
    """
    Removes texts shorter than 100 characters.
    :param df: dataframe
    :return: dataframe without short texts
    """
    df = df[df['text'].apply(len) > 100]

    if df.shape[0] < 1:
        raise Exception('Empty Dataframe')
    return df

def drop_non_english(df):
    """
    Removes texts that are not english.
    :param df: dataframe
    :return: dataframe only english texts
    """
    df = df[df['text'].apply(lambda text: detect(text[:100]) == 'en')]  
    return df


def contains_verb(sent):
    """
    Checks is the input sentence contains at least one verb.
    :param df: sentence
    :return: whether the input sentence contains a verb
    """
    try:
        doc = nlp(sent)
        return any([tok.pos_ in ['VERB','AUX'] for tok in doc])
    except:
        print(sent)
        return False


def split_into_sentences(df):
    """
    Splits the texts into sentences and only keeps sentences with a verb and with more than three words.
    :param df: dataframe
    :return: dataframe, dataframe split into sentences
    """
    df['sents'] = df['text'].apply(sent_tokenize)
    df['sents'] = df['sents'].apply(lambda sents: [sent for sent in sents if (contains_verb(sent) & (len(str(sent).split(' '))>3))])
    df = df[df['sents'].apply(len) > 0]
    df_sent = df.explode('sents')
    df_sent = df_sent.drop_duplicates(subset=['sents'])
    return df, df_sent


def create_n_sent_windows(df, window_size):
    """
    Splits the text into text segments of n sentences with overlapping sentences.
    :param df: input dataframe
    :param window_size: number of sentences per segment
    :return: dataframe of text segments
    """
    df['merged'] = df['sents'].apply(
        lambda sents: [(s, r.values) for (s, r) in zip(sents, pd.Series(sents).rolling(window_size, center=True))])
    df = df.explode('merged')
    df['sents'] = df['merged'].apply(lambda x: x[0])
    df['windowed_' + str(window_size)] = df['merged'].apply(lambda x: ' '.join(x[1]))
    df = df[['sents', 'windowed_' + str(window_size)]].drop_duplicates()
    return df


def remove_irrelevant(df):
    """
    Removes texts that do not contain a keyword and a brand.
    :param df: dataframe
    :return: dataframe of relevant texts
    """
    keywords = pd.read_csv('../data/kw_list.csv')['Keyword'].to_list()
    brands = pd.read_csv('../data/brand_list.csv')['Brand'].to_list()
    
    pattern_kws = '\\b(' + '|'.join(keywords) + ')\\b'
    df_kw = df[df['windowed_3'].str.contains(pattern_kws).fillna(False)]
    
    pattern_brands = '\\b(' + '|'.join([b for b in brands]) + ')\\b'
    df = df_kw[df_kw['windowed_3'].str.contains(pattern_brands).fillna(False)]
    return df


if __name__ == '__main__':
    # Read abstracts file
    df_abstract = pd.read_pickle('../data/df_abstracts.pckl')
    df_abstract = df_abstract[df_abstract['text'].notna()]
    df_abstract = df_abstract[df_abstract['text']!='[No abstract available]']
    
    # Reads all files in the folder to a Dataframe
    path_to_pdfs = ''
    df_pdf = read_pdfs(path_to_pdfs)
    
    # Merge abstract and pdf dataframes
    df = pd.concat([df_abstract, df_pdf])

    # clean data
    df = drop_duplicates(df)
    df = remove_special_characters(df)
    df = remove_copyright(df)
    df = drop_short_texts(df)
    df = drop_non_english(df)

    # splits the text into segments of 3 sentences
    df, df_sent = split_into_sentences(df)
    df_3 = create_n_sent_windows(df, 3)
    df = df_sent.merge(df_3, on='sents', how='left').drop_duplicates(subset='sents').drop(columns=['text'])
    df = remove_irrelevant(df)

    # save preprocessed dataframe
    df.to_pickle('../data/df_preprocessed.pckl')