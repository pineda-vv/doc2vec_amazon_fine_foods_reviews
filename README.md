

# Deriving sentiment from food52 user comments
### **by applying a doc2vec/classifier analysis using the  Amazon Fine Food Reviews**
#### Data can be found here. (https://www.kaggle.com/snap/amazon-fine-food-reviews


```python
import pandas as pd
import numpy as np
import gensim
```


```python
# Loading full dataset into pd DataFrame
df = pd.read_csv('data/Reviews.csv')
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B001E4KFG0</td>
      <td>A3SGXH7AUHU8GW</td>
      <td>delmartian</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1303862400</td>
      <td>Good Quality Dog Food</td>
      <td>I have bought several of the Vitality canned d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B00813GRG4</td>
      <td>A1D87F6ZCVE5NK</td>
      <td>dll pa</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1346976000</td>
      <td>Not as Advertised</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>B000LQOCH0</td>
      <td>ABXLMWJIXXAIN</td>
      <td>Natalia Corres "Natalia Corres"</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1219017600</td>
      <td>"Delight" says it all</td>
      <td>This is a confection that has been around a fe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>B000UA0QIQ</td>
      <td>A395BORC6FGVXV</td>
      <td>Karl</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1307923200</td>
      <td>Cough Medicine</td>
      <td>If you are looking for the secret ingredient i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B006K2ZZ7K</td>
      <td>A1UQRSCLF8GW1T</td>
      <td>Michael D. Bigham "M. Wassir"</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1350777600</td>
      <td>Great taffy</td>
      <td>Great taffy at a great price.  There was a wid...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Basic overview of numeric data
df.describe().T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>568454.0</td>
      <td>2.842275e+05</td>
      <td>1.640987e+05</td>
      <td>1.0</td>
      <td>1.421142e+05</td>
      <td>2.842275e+05</td>
      <td>4.263408e+05</td>
      <td>5.684540e+05</td>
    </tr>
    <tr>
      <th>HelpfulnessNumerator</th>
      <td>568454.0</td>
      <td>1.743817e+00</td>
      <td>7.636513e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>8.660000e+02</td>
    </tr>
    <tr>
      <th>HelpfulnessDenominator</th>
      <td>568454.0</td>
      <td>2.228810e+00</td>
      <td>8.289740e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>9.230000e+02</td>
    </tr>
    <tr>
      <th>Score</th>
      <td>568454.0</td>
      <td>4.183199e+00</td>
      <td>1.310436e+00</td>
      <td>1.0</td>
      <td>4.000000e+00</td>
      <td>5.000000e+00</td>
      <td>5.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>Time</th>
      <td>568454.0</td>
      <td>1.296257e+09</td>
      <td>4.804331e+07</td>
      <td>939340800.0</td>
      <td>1.271290e+09</td>
      <td>1.311120e+09</td>
      <td>1.332720e+09</td>
      <td>1.351210e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
from gensim.models import Doc2Vec
# from gensim.models.doc2vec import LabeledSentence  <-deprecated use TaggedDocument instead
# from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import random
```

### Taking a 100K representative sample


```python
np.random.seed(42)
idx_samp = [np.random.randint(0, 568454) for n in range(100000)]
```

### New Dataframe with 100K rows


```python
new_df = df.iloc[idx_samp, :]
```


```python
# reload this file if need to redo the analysis
# new_df.to_csv('data/amzn_subset.csv', sep='\t', index=False)

```


```python
# Comparing Score Distribution of original dataset to 100K sample
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
fig, axs = plt.subplots(2,1, figsize=(14, 8))
axs[0].hist(df['Score'], color='b')
axs[1].hist(new_df['Score'], color='r', alpha=0.2)
```




    (array([  9195.,      0.,   5363.,      0.,      0.,   7405.,      0.,
             14052.,      0.,  63985.]),
     array([ 1. ,  1.4,  1.8,  2.2,  2.6,  3. ,  3.4,  3.8,  4.2,  4.6,  5. ]),
     <a list of 10 Patch objects>)




![png](output_11_1.png)



```python

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
```


```python
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
    new_data = []
    for d in data:
        new_str = d.lower()
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))
        new_data.append(dlist)
    return new_data
```


```python
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield TaggedDocument(doc, [self.labels_list[idx]])

```


```python
# labels = list(new_df.index)
data = new_df['Text']
```


```python
new_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>121958</th>
      <td>121959</td>
      <td>B003M63C0E</td>
      <td>A27L3LYLHCQZYG</td>
      <td>Amberabcg</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1338336000</td>
      <td>Excellent Quality; my dogs LOVE it!</td>
      <td>I have 5-7 dogs at any given time, sometimes f...</td>
    </tr>
    <tr>
      <th>131932</th>
      <td>131933</td>
      <td>B000CQIDHY</td>
      <td>A1AES697PC2IW5</td>
      <td>Kevin Kiersky "oceaneagle"</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1297123200</td>
      <td>Earl Grey of Earl Grey +++</td>
      <td>I already liked regular Stash Earl Grey and so...</td>
    </tr>
    <tr>
      <th>365838</th>
      <td>365839</td>
      <td>B001EQ55MM</td>
      <td>A1Q99N7YEJ6CZJ</td>
      <td>K. makki "KID"</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1199577600</td>
      <td>Great coffee</td>
      <td>eight oclock makes great coffee and with balan...</td>
    </tr>
    <tr>
      <th>259178</th>
      <td>259179</td>
      <td>B001PICX42</td>
      <td>A3RJVINZDBOUNE</td>
      <td>N. S. Goodman</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1309910400</td>
      <td>Yummy Fruit Snacks</td>
      <td>I bought these for my kids but find myself eat...</td>
    </tr>
    <tr>
      <th>110268</th>
      <td>110269</td>
      <td>B000B6MV9Q</td>
      <td>A2LN6GJQI1S9EW</td>
      <td>Juliane J. Hass</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1325203200</td>
      <td>funnel cake mix</td>
      <td>This is very good and just like the gourmet on...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Id column used as labels
labels = new_df['Id'].values
```


```python
data = nlp_clean(data)
```


```python
it = LabeledLineSentence(data, labels)


```

### Building the doc2vec model


```python
model = Doc2Vec(vector_size=300, min_count=0, alpha=0.025, min_alpha=0.025, workers=4)
model.build_vocab(it)
```


```python
xx = model.epochs
```


```python
trained = model.train(it, total_words=model.corpus_count, total_examples=len(data), epochs=xx)
```


```python
# the trained doc2vec model can be saved - uncomment the following line to do so
# model.save("amzn_comments_gs_model")
```

### Deriving document vectors from model


```python
vectorized_comments = [model.docvecs[label] for label in labels]
```

### Refactor scores to yield 1 [score > 3] or 0 [score <= 3]


```python
new_df['Score'].value_counts()
```




    5    63985
    4    14052
    1     9195
    3     7405
    2     5363
    Name: Score, dtype: int64




```python
def adjust_score(score):
    if score <= 3:
        return 0
    elif score > 3:
        return 1
```


```python
new_df['score_adj'] = new_df['Score'].apply(adjust_score)
```

    /Applications/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.



```python
new_df['score_adj'].value_counts()
```




    1    78037
    0    21963
    Name: score_adj, dtype: int64



### *X, *y values set for classifier work


```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```


```python
X = vectorized_comments
```


```python
y = new_df['score_adj'].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

### Logistic Regression CV cross-validation and grid search optimization


```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
lr = LogisticRegression()             # initialize the model

grid = GridSearchCV(lr, param_grid, cv=12, scoring = 'accuracy', )

```


```python
grid.fit(X_train, y_train)
```




    GridSearchCV(cv=12, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)




```python
# Best 'C' parameter for Logistic Regression
grid.best_params_
```




    {'C': 1000}




```python
logic_mod = LogisticRegression(C=1000) # C value determined from GridsearchCV
```


```python
fitted = logic_mod.fit(X_train, y_train)
```


```python
fitted.score(X_test, y_test)
```




    0.8256



### Random Forest Classifier grid search optimization


```python
rfc = RandomForestClassifier(n_jobs=-1) 
 
# Use a grid over parameters of interest
param_grid2 = { 
           "n_estimators" : [10, 40, 100],
           "max_depth" : [1, 5, 10, 20, 30],
           "min_samples_leaf" : [1, 2, 6, 10]}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid2, verbose=1)
CV_rfc.fit(X_train, y_train)
print (CV_rfc.best_params_)
```

    Fitting 3 folds for each of 60 candidates, totalling 180 fits


    [Parallel(n_jobs=1)]: Done 180 out of 180 | elapsed: 62.9min finished


    {'max_depth': 30, 'min_samples_leaf': 1, 'n_estimators': 40}



```python
CV_rfc.best_params_
```




    {'max_depth': 30, 'min_samples_leaf': 1, 'n_estimators': 40}




```python
rfc_best = RandomForestClassifier(n_jobs=-1, max_depth=30, min_samples_leaf=1, n_estimators=40) 
fitted_rfc = rfc_best.fit(X_train, y_train)
```


```python
fitted_rfc.score(X_test, y_test)
```




    0.82355999999999996




```python
# pickled Classifier models after cross-val
import pickle
```


```python
# with open('log_reg_model.pkl', 'wb') as f:
#     pickle.dump(logic_mod, f)
```


```python
# with open('rfc_best.pkl', 'wb') as f:
#     pickle.dump(rfc_best, f)
```


```python
from sklearn.metrics import roc_curve
```


```python
# Building ROC curve for Logistic Regression Model
pos_probe = logic_mod.predict_proba(X_test)[:, 1]
```


```python
fpr, tpr, _ = roc_curve(y_test, pos_probe)
```


```python
# Building ROC curve for Random Forest Model
rfc_proba = fitted_rfc.predict_proba(X_test)
rfc_pos = rfc_proba[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rfc_pos)
```


```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
```

###  Building a Gaussian Naive Bayes Model, Cross-val, ROC curve


```python

nb_mod = GaussianNB()
```


```python
cv_score = cross_val_score(nb_mod, X_train, y_train)
```


```python
cv_score.mean()
```




    0.6720799999999999




```python
nb_mod.fit(X_train, y_train)
nb_pos = nb_mod.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_pos)
```

### Plotting the three different ROC curves for recall and precision.  Logistic Regression Model shows slightly better precision and recall compared to Random Forest. Naive Bayes model didn't do too well.


```python
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(fpr, tpr,color='b',alpha=0.4, label='Logistic Regression')
ax.plot(rf_fpr, rf_tpr, color='c', alpha=0.4, label='Random Forest' )
ax.plot(nb_fpr, nb_tpr, color='r', alpha=0.4, ls='-.', label='Gaussian Naive Bayes')
ax.grid(False)
ax.set_xlabel("False Positie Rate", fontweight='bold', fontsize=14)
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
ax.set_ylabel("True Positive Rate", fontweight='bold', fontsize=14)
ax.set_title("ROC Plot - Classifier Model Performance on Vectorized Comments(doc2vec)", fontweight='bold', fontsize=14)
ax.plot([0, 1], [0, 1], color='k', ls='--')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('data/doc2vec_files/roc_model_comparison')
```

    Exception ignored in: <object repr() failed>
    Traceback (most recent call last):
      File "/usr/local/Cellar/apache-spark/2.1.1/libexec/python/pyspark/ml/wrapper.py", line 76, in __del__
        SparkContext._active_spark_context._gateway.detach(self._java_obj)
    AttributeError: 'ALS' object has no attribute '_java_obj'



![png](output_63_1.png)


### Summary of Model Selection

|Model|CV Score|Optimized Parameters|
|:---|:---:|:---|
|Logistic Regression| 0.826 | C = 1000|
|Random Forest | 0.824 | 'max_depth': 30, 'min_samples_leaf': 1, 'n_estimators': 40|
|GaussianNB|0.672|N/A|

---

## Begin analysis of food52 comments


```python
# create model from full dataset
log_model_final = LogisticRegression(C=1000)
```


```python
final_fit = log_model_final.fit(X, y)
```


```python
df_comments = pd.read_csv('data/recipe_user_comment.csv')
```


```python
del df_comments['Unnamed: 0']
```


```python
df_comments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 56056 entries, 0 to 56055
    Data columns (total 3 columns):
    recipe_title    55943 non-null object
    user            55943 non-null object
    comment         47906 non-null object
    dtypes: object(3)
    memory usage: 1.3+ MB


### Separate rows that have no comment - will append later


```python
df_rec_no_comments = df_comments[df_comments['comment'].isnull()]
```


```python
df_rec_no_comments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8150 entries, 37 to 56055
    Data columns (total 3 columns):
    recipe_title    8037 non-null object
    user            8037 non-null object
    comment         0 non-null object
    dtypes: object(3)
    memory usage: 254.7+ KB



```python
df_comments.dropna(inplace=True)
```


```python
df_comments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 47906 entries, 0 to 56051
    Data columns (total 3 columns):
    recipe_title    47906 non-null object
    user            47906 non-null object
    comment         47906 non-null object
    dtypes: object(3)
    memory usage: 1.5+ MB


### Get comment text, clean, infer vectors from doc2vec model built with AMZN comment data


```python
com_f52 = [comment for comment in df_comments['comment']]
```


```python
com_f52 = nlp_clean(com_f52)
```


```python
com_f52_vectors = [model.infer_vector(com) for com in com_f52]
```

### Get probability estimates for new vector data


```python
f52_proba = log_model_final.predict_proba(com_f52_vectors)
```

### Assign Sentiment - plus == 1, negative == 0 in probabilities


```python
df_comments['sentiment_plus'] = f52_proba[:, 1]
```


```python
df_comments['sentiment_negative'] = f52_proba[:, 0]
```


```python
df_comments.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recipe_title</th>
      <th>user</th>
      <th>comment</th>
      <th>sentiment_plus</th>
      <th>sentiment_negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Austin Diner-Style Queso</td>
      <td>alex</td>
      <td>Do as the locals do: On the stovetop or in the...</td>
      <td>0.919253</td>
      <td>0.080747</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Austin Diner-Style Queso</td>
      <td>petalpusher</td>
      <td>This recipe is good for those of us who are al...</td>
      <td>0.901283</td>
      <td>0.098717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Austin Diner-Style Queso</td>
      <td>EmFraiche</td>
      <td>How much water do you add? I don’t see an amount.</td>
      <td>0.516155</td>
      <td>0.483845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Austin Diner-Style Queso</td>
      <td>Jennifer M</td>
      <td>I put in equal parts milk and water and it see...</td>
      <td>0.238458</td>
      <td>0.761542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Austin Diner-Style Queso</td>
      <td>Ceil_the_great</td>
      <td>Sounds tasty! Can you recommend a substitute f...</td>
      <td>0.216062</td>
      <td>0.783938</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rec_no_comments['sentiment_plus'] = 0
```

    /Applications/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.



```python
df_rec_no_comments['sentiment_negative'] = 0
```

    /Applications/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.



```python
df_rec_no_comments.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recipe_title</th>
      <th>user</th>
      <th>comment</th>
      <th>sentiment_plus</th>
      <th>sentiment_negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>Anna Jones' Favorite Lentils with Roasted Toma...</td>
      <td>Kristen Miglore</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Seedlip's A Good Dill</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Raspberry Habanero Relish</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Crostini with Roasted Grapes, Vanilla Yogurt, ...</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Gelato Valentino</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rebuild food52 dataset - recipes with no comments given 0 sentiment
new_comment_df = pd.concat([df_comments, df_rec_no_comments])
```


```python
new_comment_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recipe_title</th>
      <th>user</th>
      <th>comment</th>
      <th>sentiment_plus</th>
      <th>sentiment_negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Austin Diner-Style Queso</td>
      <td>alex</td>
      <td>Do as the locals do: On the stovetop or in the...</td>
      <td>0.919253</td>
      <td>0.080747</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Austin Diner-Style Queso</td>
      <td>petalpusher</td>
      <td>This recipe is good for those of us who are al...</td>
      <td>0.901283</td>
      <td>0.098717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Austin Diner-Style Queso</td>
      <td>EmFraiche</td>
      <td>How much water do you add? I don’t see an amount.</td>
      <td>0.516155</td>
      <td>0.483845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Austin Diner-Style Queso</td>
      <td>Jennifer M</td>
      <td>I put in equal parts milk and water and it see...</td>
      <td>0.238458</td>
      <td>0.761542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Austin Diner-Style Queso</td>
      <td>Ceil_the_great</td>
      <td>Sounds tasty! Can you recommend a substitute f...</td>
      <td>0.216062</td>
      <td>0.783938</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new_comment_df.to_csv('new_sentiment_comment.csv', sep='\t')
```


```python
# Dictionary to change user and recipe data into numeric - ALS model requirement
user_dict = {user: num for num, user in enumerate(list(set(new_comment_df['user'])))}
recipe_dict = {recipe: num for num, recipe in enumerate(list(set(new_comment_df['recipe_title'])))}
```


```python
def recipe_id(col):
    return recipe_dict[col]
def user_id(col):
    return user_dict[col]
```


```python
new_comment_df['recipe_id'] = new_comment_df['recipe_title'].apply(recipe_id)
new_comment_df['user_id'] = new_comment_df['user'].apply(user_id)
```


```python
# New dataframe for ALS modeling
senti2 = new_comment_df[['user_id', 'recipe_id', 'sentiment_plus']]
```


```python
# senti2.to_csv('data/doc2vec_files/cf_senti2.csv', sep='\t')
```


```python
senti2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>recipe_id</th>
      <th>sentiment_plus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5323</td>
      <td>3990</td>
      <td>0.919253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17383</td>
      <td>3990</td>
      <td>0.901283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10763</td>
      <td>3990</td>
      <td>0.516155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6589</td>
      <td>3990</td>
      <td>0.238458</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4404</td>
      <td>3990</td>
      <td>0.216062</td>
    </tr>
  </tbody>
</table>
</div>



### ALS recommender 


```python
import pyspark as ps
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import pandas as pd
import numpy as np


class RecipeRecommender():
    """
    Recommendation engine class - uses MLLib Alternating Least Squares Model
    """

    def __init__(self, rating_type='implicit', rank=5, reg=1):
        """ Initializes Spark and ALS model selection """
        spark = (
            ps.sql.SparkSession.builder
            .master('local[4]')
            .appName('BVS')
            .getOrCreate()
        )
        self.spark = spark
        self.sc = self.spark.sparkContext

        if rating_type == 'implicit':
                self.model = ALS(
                maxIter=5, 
                rank=rank,
                regParam=reg, 
                implicitPrefs=True,
                userCol="user_id", 
                itemCol="recipe_id", 
                ratingCol="compound",
                coldStartStrategy="drop"
                )
        if rating_type == 'explicit':
                self.model = ALS(
                maxIter=5,
                rank=rank,
                itemCol='recipe_id',
                userCol='user_id',
                ratingCol='rating',
                nonnegative=True,
                regParam=reg,
    #             coldStartStrategy="drop"
                )


    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'recipe', 'rating'

        Returns
        -------
        self : object
            Returns self.
        """
        """ Convert a Pandas DF to a Spark DF """
        ratings_df = self.spark.createDataFrame(ratings)
        """
        Train the ALS model. We'll call the trained model `recommender`.
        """
        self.recommender_ = self.model.fit(ratings_df)
        return self


    def transform(self, requests):
        """
        Predicts the ratings for a given set of user_id/recipe_id pairs.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        # Convert a Pandas DF to a Spark DF
        requests_df = self.spark.createDataFrame(requests)
        self.predictions = self.recommender_.transform(requests_df)
        return self.predictions.toPandas()


    def evaluate(self, requests, pred_col='prediction'):
        requests_df = self.spark.createDataFrame(requests)
        evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol=pred_col
            )
        rmse = evaluator.evaluate(requests_df)
        return rmse

    def recommend_for_all(self, who='users', number=10):
        if who == 'users':
            user_recs = self.recommender_.recommendForAllUsers(number)
            return (user_recs.toPandas())
        else:
            recipe_recs = self.recommender_.recommendForAllItems(number)
            return (recipe_recs.toPandas)

    def model_save(self, filepath):
        self.recommender_.save(self.sc, filepath)


if __name__ == "__main__":
    pass

```


```python
from sklearn.metrics import mean_squared_error
```


```python
def rmse(preds, train_df, test_df):
    train_rating_mean = train_df['compound'].mean()
    new_df = preds.merge(test_df, how='left', on=['user_id', 'recipe_id'])
    new_df['user_bias'] = new_df['user_id'].apply(user_bias)
    new_df['recipe_bias'] = new_df['recipe_id'].apply(recipe_bias)
    new_df['total_bias'] = new_df['recipe_bias'] + new_df['user_bias']
    new_df['adjusted'] = new_df['prediction'] + new_df['total_bias'] + 1.4
    new_df['adjusted'].fillna(new_df['total_bias'].apply(lambda x: train_rating_mean + x ), inplace=True)
    y_pred = new_df['adjusted']
    y_true = new_df['compound']
    return np.sqrt(mean_squared_error(y_true, y_pred)), new_df

def user_bias(col):
    user_groups = train_df['compound'].mean() - train_df.groupby('user_id')['compound'].mean()
    if col in user_groups:
        return user_groups[col]
    else:
        return 0

def recipe_bias(col):
    recipe_groups = train_df['compound'].mean() - train_df.groupby('recipe_id')['compound'].mean()
    if col in recipe_groups:
        return recipe_groups[col]
    else:
        return 0
```


```python
senti2.rename(columns={'sentiment_plus':'rating'}, inplace=True)
```

    /Applications/anaconda/envs/py36/lib/python3.6/site-packages/pandas/core/frame.py:2844: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      **kwargs)



```python
y = senti2['rating'].values
X = senti2.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)
cols = 'user_id recipe_id compound'.split()
train_df.columns = cols
test_df.columns = cols
test_df_preds = test_df[['user_id', 'recipe_id']]
```


```python
senti2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>recipe_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5323</td>
      <td>3990</td>
      <td>0.919253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17383</td>
      <td>3990</td>
      <td>0.901283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10763</td>
      <td>3990</td>
      <td>0.516155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6589</td>
      <td>3990</td>
      <td>0.238458</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4404</td>
      <td>3990</td>
      <td>0.216062</td>
    </tr>
  </tbody>
</table>
</div>




```python
rec_f52 = RecipeRecommender()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-271-1e91dd286ee6> in <module>()
    ----> 1 rec_f52 = RecipeRecommender()
    

    <ipython-input-270-2f139ce4a182> in __init__(self, rating_type, rank, reg)
         32                 itemCol="recipe_id",
         33                 ratingCol="compound",
    ---> 34                 coldStartStrategy="drop"
         35                 )
         36         if rating_type == 'explicit':


    /usr/local/Cellar/apache-spark/2.1.1/libexec/python/pyspark/__init__.py in wrapper(self, *args, **kwargs)
         99             raise TypeError("Method %s forces keyword arguments." % func.__name__)
        100         self._input_kwargs = kwargs
    --> 101         return func(self, **kwargs)
        102     return wrapper
        103 


    TypeError: __init__() got an unexpected keyword argument 'coldStartStrategy'



```python
rec_f52.fit(train_df)
```




    <__main__.RecipeRecommender at 0x171ae0828>




```python
preds = rec_f52.transform(test_df_preds)
# train_preds = rec_f52.transform(train_df_preds)
```


```python
rmse_score, new_df = rmse(preds, train_df, test_df)
# rmse_train = rmse(train_preds, train_df, train_df)
```


```python
rmse_score
```




    1.0306497301007747




```python
new_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>recipe_id</th>
      <th>prediction</th>
      <th>compound</th>
      <th>user_bias</th>
      <th>recipe_bias</th>
      <th>total_bias</th>
      <th>adjusted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15373.0</td>
      <td>148.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.663692</td>
      <td>0.000000</td>
      <td>0.663692</td>
      <td>1.327385</td>
    </tr>
    <tr>
      <th>7917</th>
      <td>4904.0</td>
      <td>6977.0</td>
      <td>-0.000013</td>
      <td>0.0</td>
      <td>0.662555</td>
      <td>-0.077679</td>
      <td>0.584876</td>
      <td>1.984862</td>
    </tr>
    <tr>
      <th>7916</th>
      <td>6381.0</td>
      <td>6726.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.592410</td>
      <td>0.663692</td>
      <td>1.256103</td>
      <td>2.656103</td>
    </tr>
    <tr>
      <th>7904</th>
      <td>15373.0</td>
      <td>4763.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.663692</td>
      <td>0.000000</td>
      <td>0.663692</td>
      <td>1.327385</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>1351.0</td>
      <td>6891.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.663692</td>
      <td>0.262656</td>
      <td>0.926349</td>
      <td>2.326349</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df.sort_values(by=['compound'], inplace=True)
```


```python
y_true = new_df['compound']
y_pred = new_df['adjusted']
x_plot = np.array(range(len(y_true)))
np.random.seed(42)
np.random.shuffle(x_plot)
fig, ax = plt.subplots(figsize=(14,8))
ax.set_title('Spark ALS Predictions versus True Score - Implicit Ratings Derived from Doc2Vec Sentiment Analysis')
ax.scatter(x_plot, y_true, s=6, alpha=0.4, label='True Score')
ax.scatter(x_plot, y_pred, s=6, c='c', alpha=0.4, label='ALS Score')
ax.legend(markerscale=3, fontsize='large')
# ax.set_xlim([2500, 12000])
# ax.set_ylim([0, 1.5])
ax.grid(False)

```


![png](output_114_0.png)



```python

```


```python

```


```python

```
