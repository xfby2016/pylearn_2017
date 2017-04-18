#-*- coding: utf-8 -*-
'''
Created on 17.4, 2017


Input:      text_data

Output:     pre_0_1

@author: xl
'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
pass
data = pd.read_csv('/Users/XFBY/Desktop/tianshan_data/Combined_News_DJIA.csv')
#print(data.head)
train_data = data[data['Date']<'2015-01-01']
test_data = data[data['Date']>'2014-12-31']
example = train_data.iloc[3,10]
#print(train_data['Label'])
example = example.lower()
#print(example)
example2 = CountVectorizer().build_tokenizer()(example)
#print(example2)
#example2_word_count = pd.DataFrame([[x,example2.count(x)] for x in set (example2)],columns=['word','count'])
#print(train.index)
train_headlines = []

for row in range(0,len(train_data.index)):
    train_headlines.append(''.join(str(x) for x in train_data.iloc[row,2:27]))
#    print(train_headlines)

basic_vectorrizer = CountVectorizer(ngram_range=(4,4))
basic_trains = basic_vectorrizer.fit_transform(train_headlines)

#print(basic_vectorrizer.get_feature_names())
print(basic_trains.shape)
#print(basic_train.toarray())

LRmodel = LogisticRegression()
lrmodel = LRmodel.fit(basic_trains,train_data['Label'])
test_headlines = []
for row in range(0,len(test_data.index)):
    test_headlines.append(''.join(str(x) for x in test_data.iloc[row,2:27]))
    pass
basic_test = basic_vectorrizer.transform(test_headlines)
prediction = lrmodel.predict(basic_test)
result = pd.crosstab(test_data["Label"],prediction,rownames=["Actual"],colnames=["Predict"])
print(result)
words = basic_vectorrizer.get_feature_names()
coeffs = lrmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'word':words,'coefficient':coeffs})
coeffdf = coeffdf.sort_values(['coefficient','word'],ascending=[0,1])
print(coeffdf