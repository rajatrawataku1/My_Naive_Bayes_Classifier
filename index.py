import pandas as pd
import scipy
import pylab
import operator
import nltk
from nltk.probability import *
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import json
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from dateutil.parser import parse
import re

train = pd.read_csv("DEV_SMS1.csv",encoding = "ISO-8859-1")

def is_float(string):
  try:
    return float(string) and '.' in string  # True if string is a number contains a dot
  except ValueError:  # String is not a number
    return False

def is_date(string):
    try:
        parse(string)
        return True
    except Exception :
        return False

def conv(sampple):
    for val in set_number:
        if val in sampple:
            return True
    return False


# feature generators function

string_input=""
set_number=set("0123456789")

def feature_generator(sample):
    string_input=sample
    string_input=string_input.lower()
    tokenize_data=word_tokenize(string_input)
    stop_eng = stopwords.words('english')
    punct = set(string.punctuation)

    # removing the stopwords
    new_tokenize_data=[]
    for sample in tokenize_data:
        if sample not in stop_eng:
            new_tokenize_data.append(sample)

    final_tokenize_data=[]
    # removing the punctuation
    for sample in tokenize_data:
        if sample not in punct:
            final_tokenize_data.append(sample)


    no_of_dates=0;
    no_of_decimal_number=0;
    no_of_normal_number=0;
    no_of_time=0;
    no_of_alphanumeric=0;
    no_money=0;
    no_link=0;
    no_of_delievered=0;
    no_of_ordered=0;


    for sample in final_tokenize_data:
        if sample.isdigit():
            no_of_normal_number=no_of_normal_number+1

        if "inr" in sample or "rs." in sample:
            no_money=no_money+1

        if 'http' in sample or 'www' in sample:
            no_link=no_link+1

        if is_float(sample):
            no_of_decimal_number=no_of_decimal_number+1;

        if is_date(sample):
            if ":" in sample:
                no_of_time=no_of_time+1
            else:
                if '-' in sample:
                    no_of_dates=no_of_dates+1

        if sample.isalnum():
            if conv(sample):
                no_of_alphanumeric=no_of_alphanumeric+1

        if 'delivered' in sample:
            no_of_delievered=no_of_delievered+1

        if 'order' in sample:
            no_of_ordered=no_of_ordered+1


    feature={}
    feature['no_of_dates']=no_of_dates
    feature['no_of_decimal_number']=no_of_decimal_number
    feature['no_of_normal_number']=no_of_normal_number
    feature['no_of_time']=no_of_time
    feature['no_money']=no_money
    feature['no_link']=no_link
    feature['aplpha_numeric']=no_of_alphanumeric

    # feature['deleivered']=no_of_delievered
    # feature['order']=no_of_ordered

    return feature

length_of_datasets=len(train)
labeled_names = [(train.iloc[i][0], train.iloc[i][1]) for i in range(length_of_datasets)]

featuresets = [(feature_generator(sms_text), label) for (label, sms_text) in labeled_names]

d=pd.DataFrame(data=featuresets)
d.to_csv("out.csv", sep=',')

train_sets=featuresets
classifier = nltk.NaiveBayesClassifier.train(train_sets)
# print(nltk.classify.accuracy(classifier, test_sets))

test= pd.read_csv("DEV_SMS1.csv",encoding = "ISO-8859-1")
index=1;
result=""
length_of_datasets_test=len(test)
ar=['RecordNo','Label']
output=pd.DataFrame(columns=ar)
for iter in range(length_of_datasets_test):
    result=classifier.classify(feature_generator(test.iloc[iter][1]))
    resultasdf=[index,result]
    output.loc[iter]=resultasdf
    index=index+1

output.to_csv("result.csv", sep=',')

# test = pd.read_csv("DEV_SMS.csv",encoding = "ISO-8859-1")
# length_of_datasets_train=len(train)
# labeled_names_train = [(train.iloc[i][0], test.iloc[i][1]) for i in range(length_of_datasets_train)]
# featuresets_train = [(feature_generator(sms_text), label) for (label, sms_text) in labeled_names_train]
