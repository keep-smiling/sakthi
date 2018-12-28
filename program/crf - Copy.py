import random
import math
import pandas as pd
import nltk 
from feature_function import FeatureFunction


def load_train_data():
    return load_data()


def load_data():
    data_list = []
    filepath = 'preprocessed_sarcastic.txt' 
    with open(filepath) as fp:  
        txt = fp.readline()
        while txt:   
            wordsList = nltk.word_tokenize(txt) 
            tagged = nltk.pos_tag(wordsList)
            #print("\n\nsample tweet\n")
            #print(txt)
            #print("\n\nPOS tag assignment : \n")
            #print(tagged)
            words = []
            labels = []
            for i in tagged:
                words.append(i[0])
                labels.append(i[1])
            #print("\n\nseparation of wprds and POS tags : \n")
            #print(words)
            #print("\n")
            #print(labels)
            #print("\n")
            data_list.append((words,labels))
            txt = fp.readline()
    return data_list
     


def create_feature_functions(train_data):
    features_functions = set()

    for words, labels in train_data:
        #print (words)
        #print (labels)
        #print("\n\nTerm pairs creation \n")
        for i in range(1, len(labels)):
            #print(labels[i-1],labels[i])
            features_functions.add(FeatureFunction(labels[i - 1], labels[i]))
    return list(features_functions)


def get_all_labels(train_data):
    available_labels = set()
    for words, labels in train_data:
        available_labels.update(labels)
    print("\n All individual POS tags\n")
    print (available_labels)
    return list(available_labels)


def initial_weights(feature_function_size):
    w=[random.random() for _ in range(feature_function_size)]
    #print (w)
    return w


def calc_empirical_expectation(feature_function, train_data):
    empirical_expectation = 0
    for words, labels in train_data:
        for i in range(1, len(labels)):
            empirical_expectation += feature_function.apply(labels[i - 1], labels[i])

    return empirical_expectation


def calc_predicted_expectation(feature_function, train_data, feature_functions,denominator):
    predicted_expectation = 0
    
    for words, labels in train_data:
        calc =calc_prob_labels_given_words(labels, feature_functions , train_data,denominator)
        for i in range(1, len(labels)):
            predicted_expectation += ( calc * sum([feature_function.apply(labels[i - 1], labels[i]) for i in range(1, len(labels))]))
    return predicted_expectation


def calc_prob_labels_given_words(labels, feature_functions, train_data,denominator):
    nominator = 1
    for j in range(1, len(labels)):
        nominator *= math.exp(sum(
            [feature_function.apply_match(labels[j - 1]) * feature_function.get_weight() for feature_function in feature_functions]))
    '''
    denominator = 1
    for words, labels in train_data:
         for j in range(1, len(labels)):
             denominator *= math.exp(sum(
                [feature_function.apply_match(labels[j - 1]) * feature_function.get_weight() for
                 feature_function in feature_functions]))
    '''
    #print("n d")
    #print(nominator,denominator)
    return nominator / denominator 

def denominator(feature_functions, train_data):
    denominator = 1
    for words, labels in train_data:
         for j in range(1, len(labels)):
             denominator *= math.exp(sum(
                [feature_function.apply_match(labels[j - 1]) * feature_function.get_weight() for
                 feature_function in feature_functions]))
    return denominator    
    

def train(train_data, all_labels, features_functions):
    learning_rate = 0.01
    iterations = 1
    weights = initial_weights(len(features_functions))
    # updates weights and count for each label paris
    for i , (feature_function, weight) in enumerate(zip(features_functions, weights)):
        v = calc_empirical_expectation(feature_function, train_data)
        feature_function.update(v,weight)
    # print each label pairs
    print("\n\nInitial Term pair weights\n")   
    for feature_function in feature_functions:
        feature_function.print_value()
    print(len(feature_functions))
    print(len(train_data))
    '''
    high = 0
    for feature_function in feature_functions:
        if high < feature_function.get_count() :
            high = feature_function.get_count()
            value = feature_function
            
    print(value)
    calc = calc_prob_labels_given_words(value,train_data)
    '''  
    
    for _ in range(iterations):
        denominator1 = denominator(feature_functions, train_data)
        i = 1
        for feature_function in feature_functions:
            #print("\n\nCurrent term pair:\n")
            #feature_function.print_value1()
            value = feature_function.get_weight()
            #print("old weight : %f " % (value))
            #print("Learning rate: %f " % (learning_rate))
            print(i)
            empirical_expectation = feature_function.get_count()
            predicted_expectation = calc_predicted_expectation(feature_function, train_data,features_functions,denominator1)
            new_weight = feature_function.get_weight() + learning_rate * (empirical_expectation - predicted_expectation)
            feature_function.put_weight(new_weight)
            #print("small variation in lemda : %f " % (-learning_rate * (empirical_expectation - predicted_expectation)) )
            #print("new weight  : %f" % (new_weight))
            #print(empirical_expectation,predicted_expectation)
            
            #print(empirical_expectation,predicted_expectation)
            #print(feature_function.get_weight())
            i += 1
    
       
    print("\n\nAfter iteration Term pair weights\n")        
    for feature_function in feature_functions:
        feature_function.print_value()  
    for feature_function in feature_functions:
        feature_function.print_value_file()  
    
if __name__ == '__main__':

    train_data = load_train_data()
    feature_functions = create_feature_functions(train_data)
    all_labels = get_all_labels(train_data)
    train(train_data, all_labels, feature_functions)
