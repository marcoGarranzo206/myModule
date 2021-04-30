from wordcloud import WordCloud
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def plot_hist_func(func,data,title,hist_kwargs = {}, legend_kwargs = {}):

    stats = list(map(func, data))
    ax = sns.histplot(stats, **hist_kwargs )
    _ = ax.set_title(title)
    median = np.median(stats)
    _  = plt.axvline(x = median, c = "red", label = f"median: {median}")
    _ = plt.legend(**legend_kwargs)

def show_wordcloud(data, stopwords):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(data)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

def plot_names_hist(list_names, title, bar_kwargs = {}, title_kwargs = {}, norm = False, top_n = None):
    
    counts = Counter(list_names)
    names = np.array([c for c in counts.keys() ])
    values = np.array(list(counts.values()))
    if norm:
        
        values = 100*values/np.sum(values)
                
    indexes = np.argsort(values)[::-1]
    if top_n is not None:
        
        indexes = indexes[:top_n]

    _ = sns.barplot(y = names[indexes], x = values[indexes], **bar_kwargs)
    _ = plt.title(title, **title_kwargs)
