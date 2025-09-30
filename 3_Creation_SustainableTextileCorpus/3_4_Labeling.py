import pandas as pd
import requests
from flashtext import KeywordProcessor
import re

def topics_in_text(text, label2id, kws):
    """
    Checks which keywords appear in the text and sets the corresponding label to 1.
    
    :param text: text
    :param label2id: dictionary that converts a label to its id
    :param kws: keyword list
    :return: list of labels
    """
    list_size = max(label2id.values()) + 1
    labels = [0] * list_size
    for kw in kws['Keyword'].to_list():
        if len(re.findall('\\b(' + kw + ')\\b', text)) > 0:
            label = kws[kws['Keyword'] == kw]['Label'].iat[0]
            labels[label2id[label]] = 1
    return labels

def keyword_classification(df):
    """
    Classifies the texts based on the keywords that appear in the text.
    
    :param df: DataFrame
    :return: keyword classifications
    """
    kws = pd.read_csv('../data/kw_list.csv')
    kws['topic_id'], unique_labels = kws['Label'].factorize()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    return df['windowed_3'].apply(lambda text: topics_in_text(text, label2id, kws))


def create_sample(to_sample, size):
    """
    Returns a sample with a similar distribution to the classes but at least 2 per class.
    
    :param to_sample: DataFrame, dataset to sample from
    :param size: int, number of samples to generate
    :return: DataFrame, sampled dataset
    """
    exploded = to_sample[['index','multi_label']].explode('multi_label')
    nrows = len(exploded)
    sample = exploded.groupby('multi_label').apply(lambda x: x.sample(max(2, int((x.count()/nrows)*size)))).reset_index()
    sample = to_sample[to_sample['index'].isin(sample['index'].unique())]
    sample = sample[['index', 'windowed_3']].rename(columns={'windowed_3': 'text'})
    return sample


def authenticate_and_get_session(username, password, server):
    """
    Authenticates and returns a session for API requests.
    
    :param username: str, API username
    :param password: str, API password
    :param server: str, server URL
    :return: requests.Session, authenticated session object
    """
    response = requests.post(f"{server}auth/token/login/", json={"username": username, "password": password})
    token = response.json()['key']
    session = requests.session()
    session.headers = {'Authorization': f'Token {token}'}
    return session

def process_suggestions(session, api_base):
    """
    Generates tagging suggestions (the keyword and brand lists) and adds them to LightTag to help annotations.
    
    :param session: requests.Session, authenticated session
    :param api_base: str, base URL of the API
    :return: list, generated suggestions
    """
    schemas = session.get(f'{api_base}projects/default/schemas/').json()
    schema = schemas[0]
    tags = session.get(f"{schema['url']}tags/").json()
    tag_map = {tag["name"]: tag["id"] for tag in tags}
    
    kws = pd.read_csv('../data/kw_list.csv')['Keyword'].to_list()
    brands = pd.read_csv('../data/brand_list.csv')['Brand'].to_list()
    
    keyword_processor = KeywordProcessor(case_sensitive=True)
    for kw in kws:
        keyword_processor.add_keyword(kw, tag_map["Keyword"])
    for brand in brands:
        keyword_processor.add_keyword(brand, tag_map["Brand"])
    
    examples = session.get(f'{api_base}projects/default/datasets/sample_1/examples/').json()
    examples += session.get(f'{api_base}projects/default/datasets/sample_2/examples/').json()
    
    suggestions = []
    for example in examples:
        for tag_id, start, end in keyword_processor.extract_keywords(example['content'], span_info=True):
            suggestions.append({"example_id": example['id'], "tag_id": tag_id, "start": start, "end": end})
    
    data = {
        "model":{"name":"suggestions", "metadata": ""},
        "suggestions":suggestions
    }
    session.post(f'{api_base}/projects/default/schemas/project_name/models/bulk/',json=data)


def preprocess_classification(df):   
    """
    Processes classification labels from labeled data.
    
    :param df: DataFrame, raw classification data
    :return: DataFrame, processed classification data
    """
    df = df.drop(columns=['annotations']).explode('classifications').dropna(subset=['classifications'])
    df['classifications'] = df['classifications'].fillna(0).apply(lambda x: x['classname'] if x != 0 else None)
    df['index'] = df['metadata'].apply(lambda x: x['index'])
    return df.drop(columns=['metadata'])

def load_and_process_annotations(filepaths):
    """
    Loads and processes annotation data from JSON files from lighttag (after labeling).
    
    :param filepaths: list, list of file paths to JSON annotation files
    :return: DataFrame, processed annotation data
    """
    dfs = []
    for path in filepaths:
        df = pd.read_json(path, orient='index')
        df = pd.DataFrame(df.T['examples'][0])
        df = df.drop(columns=['seen_by', 'comments', 'example_id']).dropna(subset=['classifications'])
        dfs.append(preprocess_classification(df))
    
    df = pd.concat(dfs).groupby('content').agg(list).reset_index()
    df = df.rename(columns={'classifications': 'multi_label', 'content': 'windowed_3'})[['windowed_3', 'multi_label']]
    df['multi_label'] = df['multi_label'].apply(lambda x: list(set(x)))
    df = pd.concat([df, pd.get_dummies(df['multi_label'].explode()).groupby(level=0).sum()], axis=1).drop(columns=['multi_label'])
        
    df = df.drop_duplicates(subset=['windowed_3'])
    df = df[df['windowed_3'].apply(len) > 1]
    df = df[df['windowed_3'].apply(len) <= 1000]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.fillna(0)
    return df

    
if __name__ == "__main__":
    # Create keyword predictions
    df = pd.read_pickle('../data/df_preprocessed.pckl')
    df['multi_label'] = keyword_classification(df)
    
    # Create data samples to be labelled
    sample_1 = create_sample(df, 300)
    sample_1.to_csv('../data/sample_1.csv')
    sample_2 = create_sample(df[~df['index'].isin(sample_1['index'])], 300)
    sample_2.to_csv('../data/sample_2.csv')
    
    # Prepare annotation for lighttag
    session = authenticate_and_get_session(username='', password='', server='')
    process_suggestions(session, api_base='')
    
    # Read annotations from lightag
    annotation_files = ['../data/sample_1_annotations.json', '../data/sample_2_annotations.json']
    df_annotated = load_and_process_annotations(annotation_files)
    df_annotated.to_csv('../data/df_annotated.csv', index=False)
