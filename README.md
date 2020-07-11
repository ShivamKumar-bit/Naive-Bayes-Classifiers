# Naive Bayes Classifier
It is a collection of algorithms based on Bayes Theorem.
To start, let us consider dataset

![datframe](https://user-images.githubusercontent.com/67604006/87218523-294f3900-c371-11ea-977a-551846ce099d.png)

The most fundamental assumption in Naive Bayes is :
- all features are independent of each other 
- and all have equal contribution to the output <br>
Though features in our data frame maybe dependent we consider them as independent.

## Bayes Theorem
The mathematical formula is given by:

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7777aa719ea14857115695676adc0914_l3.svg)

It's the probability of an event say A occurring given the probability of another event say B that has already occurred. 

With reference to our dataset it can be written as:

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e85875a7ff9e9b557eab6281cc7ff078_l3.svg)

Here y is our target variable given that X has already occured.

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-1c3f5ab570cf0ab3f43d5c18c645b67a_l3.svg)
 
X= x1,x2,x3...... all these are features of our dataset.

The above can be expressed as :

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8171c1fe2cbd3ed62bc3f40d682c0512_l3.svg)

Since the denominator is same for any given input ,so

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c778553cb5a67518205ac6ea18502398_l3.svg)

Now, we need to create a classifier model. For this, we find the probability of given set of inputs for all possible values of our target variable y and pick up the output with maximum probability. This can be represented as :

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f3637f468262bfbb4accb97da8110028_l3.svg)

So we need to find two things :
- P(y) 
- P(xi/y)

We need to find P(xi/yj) for each xi in X and yj in y. All these calculations have been demonstrated in the tables below:

![Image](https://media.geeksforgeeks.org/wp-content/uploads/naive-bayes-classification.png)

Let us say we have new test data 

```ruby
today = (Sunny, Hot, Normal, False)
```
Probabilty of playing golf today:

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c6067bf0bf53532b6701c72215bc0758_l3.svg)

Probabilty of not playing golf today:

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ed23967bcb3871bd6919752aa396a167_l3.svg)

Since, P(today) is common in both probabilities, we can ignore it and find proportional probabilities as:

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e061a86d4158d65787e64c4cdfd15f17_l3.svg)

and

![Image](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-176cc113842cb9f7bf3e645e10381bec_l3.svg)

Since P(Yes/today) > P(No/today)

So, prediction that golf would be played is ‘Yes’.


**Bernoulli Naive Bayes** : It assumes that all our features are binary such that they take only two values. Means 0s can represent “word does not occur in the document” and 1s as "word occurs in the document" .

**Multinomial Naive Bayes** : It's is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain frequency to represent). In text learning we have the count of each word to predict the class or label.

**Gaussian Naive Bayes** : Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features are continuous. For example in Iris dataset features are sepal width, petal width, sepal length, petal length. So its features can have different values in data set as width and length can vary. We can’t represent features in terms of their occurrences. This means data is continuous.

# Let's Code-
```ruby
import pandas as pd
df = pd.read_csv('spam.csv')
df.head()
```

![Image](https://user-images.githubusercontent.com/67604006/87219859-82709a00-c37c-11ea-92cb-c747569a5a2b.png)

```ruby
df.groupby('Category').describe() # grouping by category
```

![p2](https://user-images.githubusercontent.com/67604006/87219862-83a1c700-c37c-11ea-8755-8f6566e54ea1.png)

```ruby
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0) #if spam then 1 else 0
df.head() 
```

![p3](https://user-images.githubusercontent.com/67604006/87219864-856b8a80-c37c-11ea-92b9-785fc75f8d1d.png)

```ruby
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam)
```

```ruby
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]    # using count vectorizer  what it is identify unique words in each observation
```
![p4](https://user-images.githubusercontent.com/67604006/87219868-87cde480-c37c-11ea-9826-0b4813378e82.png)

For reference you can refer [CountV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

```ruby
from sklearn.naive_bayes import MultinomialNB # multimonial classifier
model = MultinomialNB()
model.fit(X_train_count,y_train)
```

```ruby
X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)
```
Gives the model score

```ruby
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
```
Using pipeline here ,for reference visit [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

```ruby
clf.score(X_test,y_test)
```
CLF score

```ruby
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

clf.predict(emails)
```
Predicting based on input 'emails' .

That's all .Thank You!

