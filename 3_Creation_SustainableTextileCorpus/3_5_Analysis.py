import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud

def generate_wordcloud(df, output_path):
    """
    Generate a word cloud image from noun phrases.
    
    :param df: DataFrame containing the texts.
    :param output_path: Path to save the generated word cloud image.
    """
    df['nps'] = df['windowed_3'].apply(lambda text: TextBlob(text).noun_phrases)
    df['nps'] = df['nps'].apply(lambda nps: [np for np in nps if len(np) > 5])
    counts = df['nps'].explode().value_counts().head(50).reset_index()
    counts = counts[~counts['nps'].isin(['furthermore', 'according', 'figure', 'accord', 'additionally', '< https'])]
    
    wordcloud = WordCloud(background_color="white", max_words=50, max_font_size=70, width=800, height=400)
    wordcloud.generate_from_frequencies(counts.set_index('nps')['Ccunt'])
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()

def plot_brands_frequency(df, output_path):
    """
    Generate a bar chart of the 30 most common brands.
    
    :param df: DataFrame containing sentences.
    :param output_path: Path to save the plot.
    """
    brands = pd.read_csv('../data/brand_list.csv')['Brand'].to_list()
    pattern_brands = '\\b(' + '|'.join(brands) + ')\\b'
    word_counts = Counter(re.findall(pattern_brands, ' '.join(df['windowed_3'].to_list())))
    
    data = pd.DataFrame(word_counts.most_common(30), columns=['Brand', 'Count'])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Brand'], y=data['Count'], orientation='v', marker_color='#91CBC8'))
    fig.update_xaxes(showgrid=False, tickangle=310, color='#243664')
    fig.update_yaxes(showgrid=False, color='#243664')
    fig.update_layout(title='30 Most Common Brands', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      width=800, height=500, font_family="Arial")
    fig.write_image(output_path)

def plot_class_distribution(df, output_path):
    """
    Generate a bar chart of the class distribution.
    
    :param df: DataFrame containing class data.
    :param output_path: Path to save the plot.
    """
    data = df.iloc[:, 1:].sum().sort_values(ascending=False)
    data_df = data.reset_index()
    data_df.columns = ['Category', 'Value']
    data_df['Category'] = data_df['Category'].apply(lambda x: x.replace("organize &", "organize &<br>"))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data_df['Category'], y=data_df['Value'], orientation='v', marker_color='#91CBC8'))
    fig.update_xaxes(showgrid=False, tickangle=310, color='#243664')
    fig.update_yaxes(showgrid=False, color='#243664')
    fig.update_layout(title='Class Distribution', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      width=1000, height=500, font_family="Arial")
    fig.write_image(output_path)
    
if __name__ == "__main__":
    df = pd.read_pickle('../data/df_annotated.pckl')
    generate_wordcloud(df, "../figures/wordcloud.png")
    plot_brands_frequency(df, "../figures/brands_bar_plot.png")
    plot_class_distribution(df, "../figures/class_distribution.png")

