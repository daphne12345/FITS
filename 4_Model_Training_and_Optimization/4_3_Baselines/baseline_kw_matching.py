import pandas as pd
from sklearn.metrics import classification_report, f1_score
import re
from Data_Preparation import prepare_data

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



if __name__ == '__main__':
    df, df_train, df_test = prepare_data()

    y_test = df_test.iloc[:, 1:].to_numpy().astype(int).tolist()
    
    # Do keyword classification
    kws = pd.read_csv('../data/kw_list.csv')
    kws['topic_id'], unique_labels = kws['Label'].factorize()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    df_test['y_pred'] = df_test['windowed_3'].apply(lambda text: topics_in_text(text, label2id, kws))
    y_pred = df_test['y_pred'].tolist()

    # Print the classification report and F1-score
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Calculate F1-score (Macro Average)
    f1_score(y_test, y_pred, average='weighted', zero_division=0)
