# What you should (and shouldn't) name your children

Every year there are numerous publications over the most popular baby names. But have you ever seen any publication covering the "flip side"? The worst names to give your child? The goal of this analysis will be to see if there is any relationship between a person's first name and how much they were liked (or disliked) at the time of death.   

### Importing Packages


```python
import requests
from bs4 import BeautifulSoup
import time
import seaborn as sns
import pandas as pd
import wordcloud
#from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     /Users/mitchfairweather/nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/mitchfairweather/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/mitchfairweather/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!





    True



### Functions used to scrape obituary data


```python
def obituaries(url):
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15'}
    r = requests.get(url, timeout = 120, headers = headers)
    r.close()
    soup = BeautifulSoup(r.content, "html.parser")
    obit_name = soup.find(class_ = "MuiTypography-root MuiTypography-h1 css-bpbdsz").get_text()
    obit_text = list(soup.find_all("p", class_ = "MuiTypography-root MuiTypography-body1 css-1yzfdrp css-9l3uo3"))
    for i in range(len(obit_text)):
        obit_text[i] = obit_text[i].get_text()
    obit_text = "".join(obit_text)
    time.sleep(.5)
    return obit_name, obit_text

    
```


```python
def get_obit_links(pages):
    obituary_links = []
    for page in tqdm(range(1, pages + 1)):
        try: 
            headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15'}
            url = "https://www.indystar.com/obituaries?page=" + str(page)
            r = requests.get(url, timeout = 120, headers = headers)
            r.close()
            soup = BeautifulSoup(r.content, "html.parser")
            links = soup.find_all("a", class_ = "link")
            for link in links:
                obituary_links.append("https://www.indystar.com" + link.get('href'))
            time.sleep(.5)
        except: 
            continue
            
    return obituary_links
            
        
```

### Scraping Obituary Data

Data for this analysis has been pulled the Indianapolis Star, the local publication serving Indianapolis, Indiana area. The oldest data published is from Dec 07, 2018. I have scraped all obituaries published in the Indy Start since that date. 


```python
obituary_links = get_obit_links(pages = 1370)
```

    100%|███████████████████████████████████████| 1370/1370 [15:55<00:00,  1.43it/s]



```python
with open("/Users/mitchfairweather/Obituaries/IndyStarObituariesLinks.csv", 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(obituary_links)
```


```python
obituary_names = []
obituary_texts = []
new_obit_links = []
bad_links = []
```


```python
for link in tqdm(obituary_links):
    for i in range(3): 
        try: 
            name, text = obituaries(str(link))
            obituary_names.append(name)
            obituary_texts.append(text)
            new_obit_links.append(link)
        except: 
            if i == 2: 
                bad_links.append(str(link))
                break 
            else: 
                continue
        else: 
            break
```

    100%|███████████████████████████████████| 27366/27366 [5:18:05<00:00,  1.43it/s]



```python
obituary_df = pd.DataFrame()
obituary_df["Names"] = obituary_names
obituary_df["Obituary"] = obituary_texts
obituary_df["Links"] = new_obit_links
```


```python
obituary_df.drop_duplicates(inplace = True)
obituary_df.reset_index(drop = True, inplace = True)
```


```python
#obituary_df.to_csv("/Users/mitchfairweather/Obituaries/IndyStarObituaries.csv")
```


```python
obituary_df.head()
```


```python
obituary_df.describe()
```

### Cleansing Data

The Names of each obituary contains various preceeding titles that we don't want to consider a "first name" including "Mr.", "Miss", etc. These are all removed as part of the creation of the "First Name" field. 


```python
obituary_df["First Name"] = obituary_df["Names"]

#Remove all text within parentheses and quotes which would indicate a nickname
first_name = []
for i in range(0, len(obituary_df)):
    name = re.sub(r'\([^)]*\)', '', obituary_df['First Name'][i])
    name = re.sub(r'\"[^)]*\"', '', obituary_df['First Name'][i])

    first_name.append(name)

obituary_df["First Name"] = first_name

obituary_df["First Name"] = obituary_df["First Name"].str.replace("Mr. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Mr ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Mrs. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Mrs ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Ms. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Ms ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Miss. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Miss ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Dr. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Dr ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Professor ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Sr. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Sr ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Jr. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Jr ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Fr. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Fr ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Lt. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Lt ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Ltc. ", '')
obituary_df["First Name"] = obituary_df["First Name"].str.replace("Ltc ", '')


obituary_df["First Name"] = obituary_df["First Name"].str.lstrip()

obituary_df["First Name"] = obituary_df["First Name"].str.split(' ', expand=True)[0]
```


```python
obituary_df.head(10)
```

The next step in cleansing the obituary data is to remove any non-numerical text and lemmatize the obituary text. 

I am going to remove non-numerical data as for the most part, the only numerical data in an obituary will be for the date the given person would have been born and subsequently passed. As such, this information provides zero value to the overal sentiment of the obituary. 

The last step for cleansing is to lemmatize the obituary text and remove all stop words. This is a fairly standard process when conducting an NLP problems. Related to stemming, lemmatization will convert each word to its base form of all its inflectional forms. 


```python
obituary_clean = []
text = ""
for i in range(0, len(obituary_df)):
    text = re.sub('[^a-zA-Z]', ' ', obituary_df['Obituary'][i])
    text = text.lower()
    text = text.split()
    
    wl = WordNetLemmatizer()
    text = [wl.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    obituary_clean.append(text)
    
obituary_df["Obituary_Clean"] = obituary_clean
```

#### Removing Obituaries with No Significant Data

As you can see, there is a significant portion of data that has little to no text in the obituary. These are typically obituaries that only have the date of passing, and sometimes the location of the funeral home where services will be held. Some have zero text published. For the purposes of our analysis, we will not want to include these and thus are removing them from our data set. 


```python
obituary_df['Length'] = obituary_df['Obituary_Clean'].str.count(' ') + 1
```


```python
plt.hist(obituary_df['Length'],bins=50,color='b')
plt.grid(True)
plt.xticks(np.arange(0, max(obituary_df.Length), 50))
plt.xlabel("Count of Words in Obituary")
plt.ylabel("Number of Obituaries")
plt.show()
```


```python
obituary_df[obituary_df['Length'] < 10]
```


```python
obituary_df = obituary_df[obituary_df['Length'] >= 10].reset_index(drop = True)
```

# Exploring the Data

Although the data is limited, its not surprising to see the most popular names in our obituary data set as names like "James", "Robert", or "Mary". I am only slightly surprised to not see "John" have a larger presence. Apparently there have not been too many Johns that have passed since 2018 in Indianapolis. 


```python
wc = WordCloud(background_color = 'white', width = 800, height = 500).generate(' '.join(obituary_df['First Name']))
plt.axis("off")
plt.imshow(wc)

```

After removing the obituaries that had little to no content, we are left with the majority of our data being still fairly small obituaries. A more uniform distribution of length would be nice to see, but it is interesting that outside of very short obituaries, the second most common length is in the range of 100-125 words. 


```python
plt.hist(obituary_df['Length'],bins = 100,color='b')
plt.grid(True)
plt.xticks(np.arange(0, max(obituary_df.Length), 50))
plt.xlabel("Count of Words in Obituary")
plt.ylabel("Number of Obituaries")
plt.show()
```

### Sentiment Analysis


```python
# initiate an analyzer
sia = SentimentIntensityAnalyzer()
```


```python
results = []
# iterate through each sentence in corpus
for i in range(len(obituary_df)):
    
    # analyze the sentiment. ss is a dictionary
    ss = sia.polarity_scores(obituary_df["Obituary_Clean"][i])
    results.append(ss)
```


```python
obituary_df = pd.concat([obituary_df, pd.DataFrame(results)], axis=1, join="inner")
obituary_df.head()
```

Not surprisingly, there is essentially zero obituaries that are viewed as fully "negative". When you consider that if a person was not well liked during their life, they either wouldn't have an obituary at all or it would be more neutral, rather than negative. Not many people will want to talk negatively about somebody who just passed, if they are discussing them at all. 


```python
plt.hist(obituary_df.compound, alpha=0.1, label='Compound', color = "blue")
plt.hist(obituary_df.neg, alpha=0.6, label='Negative', color = "pink")
plt.hist(obituary_df.neu, alpha=0.6, label='Neutral', color = "grey")
plt.hist(obituary_df.pos, alpha=0.6, label='Positive', color = "lightgreen")
plt.legend(loc='upper left')
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()
```

There are a a handful of obituaries that have a low compound score. Lets take a look at what the sentiment analysis labeled as the most negative and most positive obituaries in our data set. 


```python
obituary_df[obituary_df.compound == min(obituary_df.compound)]
```

At first glance, this reads as though it is a very standard and basic obituary. It highlights just the basics of what you would expect to see and nothing much more. That makes sense considering the sentiment scores the model gave it, where it is almost entirely Neutral. It is neither good nor bad. 


```python
print(obituary_df[obituary_df.compound == min(obituary_df.compound)].Obituary.values[0])
```


```python
obituary_df[obituary_df.compound == max(obituary_df.compound)]
```

On the opposite end of the spectrum, we see below what is marked as the obituary with the highest compound score. In comparison to the most negative obituary, it is no surprise why this one is viewed so highly. Between the length, level of detail, and frequent use of very positive, detailed language, it makes sense how this would be the most highly ranked obituary.


```python
print(obituary_df[obituary_df.compound == max(obituary_df.compound)].Obituary.values[0])
```

### Exploring relathionships in the data


```python
summaryTable = obituary_df.groupby('First Name').agg(Average=('compound', 'mean'), Count=('First Name', 'count')).reset_index()
```

Yet another challenge for this analysis is the number of times a given name appears in the dataset. Of the roughly 2700 unique names in our dataset, nearly 1600 appear only once. About 75% of the names in our data set we have fewer than 4 obituaries for. However, that still leaves us with about 700 names in our dataset that appear at least 4 times. In order to get some meaningful insights, we will remove obituaries for names that appear fewer than 4 times in our data. 


```python
plt.hist(summaryTable['Count'],range=[0, 50], bins = 100,color='b')
plt.grid(True)
plt.xticks(np.arange(0, 50, 2))
plt.yticks(np.arange(0, 1700, 100))

plt.xlabel("Count of Words in Obituary")
plt.ylabel("Number of Obituaries")
plt.show()
```


```python
topNames = summaryTable[summaryTable.Count >= 4].nlargest(40, "Average").reset_index(drop = True).sort_values("Average", ascending=False)
bottomNames = summaryTable[summaryTable.Count >= 4].nsmallest(40, "Average").reset_index(drop = True).sort_values("Average", ascending=True)
```

There is VERY little variation in the top names. All have a score of essentially 1, and unfortunately there is only a single observation for each name. That doesn't really tell us much of anything, other than that particular Del was extremely well liked. 


```python
fig, ax1 = plt.subplots(figsize=(9, 6))

# Instantiate a second axes that shares the same x-axis
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis="x", labelrotation = 75)
ax1.tick_params(axis = "y", labelcolor="dodgerblue")
ax1.set_ylabel("Average Compound Score (higher the better)", fontsize=10, color = "dodgerblue")

ax2 = ax1.twinx()  
ax2.tick_params(axis = "y", labelcolor="crimson")
ax2.set_ylabel("Count of Names", fontsize=10, color = "crimson") 

ax2.set_ylim(0, max(topNames.Count)+5)

ax1.bar(topNames['First Name'],topNames['Average'], color = "dodgerblue", alpha = .5)
ax1.set_xticklabels(topNames['First Name'], fontsize=8)

ax2.plot(topNames['First Name'],topNames['Count'], color = "crimson")
fig.suptitle("Indiana's Most Liked Names", fontsize=20)

```

Unlike the top ranked obituaries, there is quite a bit more variation in the bottom ranked obituaries. The difference between the worst name on average is sizeable compared to the 40th worst. However, the issue persists of very little data. Only a handful of these names have more than a single occurence in the data set. If we were to make any conclusions about a relation between a person's name and how well they were liked, we would need significantly more data. 


```python
fig, ax1 = plt.subplots(figsize=(9, 6))

# Instantiate a second axes that shares the same x-axis
ax1.set_ylim(-1, 0)
ax1.tick_params(axis="x", labelrotation = 75)
ax1.tick_params(axis = "y", labelcolor="dodgerblue")
ax1.set_ylabel("Average Compound Score (higher the more liked)", fontsize=10, color = "dodgerblue")


ax2 = ax1.twinx()  
ax2.tick_params(axis = "y", labelcolor="crimson")
ax2.set_ylabel("Count of Names", fontsize=10, color = "crimson") 

ax2.set_ylim(0, max(topNames.Count)+5)

ax1.bar(bottomNames['First Name'],bottomNames['Average'], color = "dodgerblue", alpha = .5)
ax1.set_xticklabels(bottomNames['First Name'], fontsize=8)

ax2.plot(bottomNames['First Name'],bottomNames['Count'], color = "crimson")
fig.suptitle("Indiana's Most Liked Names", fontsize=20)


```

## K Means Clustering

The final area of exploration for our obituary data set is through clustering. The goal of using this algorithm is to identify groups in our data. Since the overarching idea of our analysis is to cluster our data into two groups, positive or negative, I have set the number of clusters to create as 2. The algorithm clusters the data on the basis of high similarity points in one cluster and low similarity points in the other cluster. 


```python
vectorizer = CountVectorizer()
vectorizer.fit(obituary_df["Obituary_Clean"])
vector = vectorizer.transform(obituary_df["Obituary_Clean"])

model = KMeans(n_clusters=2, max_iter=300, random_state=True, n_init=10)
fit = model.fit_predict(X=vector)

obituary_df["Cluster"] = fit
```

The first plot I want to look at is the difference between the positive scores in each cluster. Interestingly, it is evident there is a significant difference between the positive scores in cluster 1 than in cluster 0. However, this must be taken with a grain of rice considering the previous issues with the data that we have highligted. 


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster',y='pos',data=obituary_df, palette='rainbow')
plt.title("Positive Score by Cluster")

```

Not surprisingly, there is essentially no difference between clusters when looking at the negative scores. I highlighted earlier that there really was no negative obituaries, so this makes sense. 


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster',y='neg',data=obituary_df, palette='rainbow')
plt.title("Negative Score by Cluster")

```

We found earlier in the analysis that there really aren't any "negative" obituaries. It was more of a battle between positive and neutral. Knowing this, it is very interesting that whereas cluster 1 was significantly higher in terms of the positive score, cluster 0 is fairly significantly different in terms of the neutral score. I would have expected a slightly larger difference, but a noticeable difference is still something to take note of. 


```python
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster',y='neu',data=obituary_df, palette='rainbow')
plt.title("Neutral Score by Cluster")

```

The last plot to look at is the difference between the compound score in each cluster. The most concerning/noticeable observation is the large amount outliers in cluster 1. The distribution is so tight, the box plot is almost not even visible. Although there is a significant difference between cluster 0 and cluster 1 in terms of the compound score, I am not sure how confident we can be in this method to take concrete conclusions from. 


```python
plt.figure(figsize=(5,10))
sns.boxplot(x='Cluster',y='compound',data=obituary_df, palette='rainbow')
plt.title("Compound Score by Cluster")

```


```python
plt.scatter(obituary_df[obituary_df.Cluster == 1]["Length"] , obituary_df[obituary_df.Cluster == 1]["compound"] , color = 'red', alpha = .5, label = "Cluster 1")
plt.scatter(obituary_df[obituary_df.Cluster == 0]["Length"] , obituary_df[obituary_df.Cluster == 0]["compound"] , color = 'grey', alpha = .5, label = "Cluster 0")

plt.xlabel("Length of Obituary (words)")
plt.ylabel("Compound Score")
plt.legend(loc='lower right')
plt.title("Length of Obituary vs Compound Score")
plt.show()

```

## Conclusions

Should you use this analysis to determine what to name and what not to name your child? Of course not. However, in the event you between two names, maybe this fun project will help you make that decision. Let's take a look at the lowest and highest rates names in Indianapolis based off of a sentiment analysis on obituaries. 

#### Most Disliked Names 

Apparently Indiana is not a fan people with names of Luther, Andre, or Roosevelt that passed away from 2018-2023. If you're thinking of naming your child one of the names in the word cloud below, maybe think twice. 


```python
bottomNames['Average'] = bottomNames['Average']*-1
wc = WordCloud(background_color = 'white', width = 800, height = 500).generate_from_frequencies(frequencies=bottomNames.set_index('First Name').to_dict()['Average'])
plt.axis("off")
plt.imshow(wc)
```

#### Indiana's most Liked Names

Are you thinking of naming your child Caroline? Jordan? What about Marianna? If so, you get two thumbs up from Indianapolis residents. 


```python
topNames['Average'] = topNames['Average']
wc = WordCloud(background_color = 'white', width = 800, height = 500).generate(' '.join(topNames['First Name']))
plt.axis("off")
plt.imshow(wc)
```
