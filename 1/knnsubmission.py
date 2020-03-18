import numpy as np
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import math
from tqdm import tqdm
from sklearn.model_selection import KFold

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
remove_digits_map = dict((ord(char), None) for char in string.digits)

# Hyper-params for grid search
k_neighbors = [1,3,5,7,9,11]

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def tokenize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map).translate(remove_digits_map)))

def getallcorpus(train_input, test_input):
    corpus = []
    paths = []
    for key in train_input:
        corpus.append(open(key, 'r').read())
        paths.append(key)
    savelentrain = len(corpus)
        
    for i in range(len(test_input)):
        corpus.append(open(test_input[i], 'r').read())
        paths.append(test_input[i])
    
    savelentest = len(corpus)- savelentrain

    return corpus, paths, [savelentrain, savelentest]

def pdrow2nparr(df, key):
    return df.loc[key].to_numpy().reshape(1, -1)

def getclass(classdict, key):
    return classdict[key]

def cosine_sim(df, key1, key2):
    e1 = pdrow2nparr(df, key1)
    e2 = pdrow2nparr(df, key2)
    sim = cosine_similarity(e1,e2)[0][0]
    
    return sim

def preprocess(train_input, test_input):
    train_dict = {}
    test_list = []
    
    train_file = open(str(train_input), "r")
    test_file = open(str(test_input), "r")

    for line in train_file:
        train_dict[line.strip().split()[0]] = line.strip().split()[1]

    for line in test_file:
        test_list.append(line.strip())

    train_file.close()
    test_file.close()

    return train_dict, test_list

def checkinhyperparam(a, hp_grid):
    for i in hp_grid:
        if a==i:
            return True
    return False

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def getlabels(train_set):
    lb = []
    for key in train_set:
        lb.append(train_set[key])
    return lb

# Cross-validation to tune hyper parameters
def KNN_fit(df, train_set, val_set_dict, k_grid):
    val_set = list(val_set_dict.keys())
    distance = {}
    valclass = {}
    
    for i in tqdm(range(len(val_set))):
        distance[val_set[i]]=[]
        valclass[val_set[i]]=[]
        for key in train_set:
            score = cosine_sim(df, key, val_set[i])
            distance[val_set[i]].append((key,score))
            
        distance[val_set[i]].sort(reverse=True, key=lambda kv: kv[1])
        
        neighbor_classes = {}
        for j in range(k_neighbors[-1]):
            if train_set[distance[val_set[i]][j][0]] in neighbor_classes:
                neighbor_classes[train_set[distance[val_set[i]][j][0]]] += 1/(j+1)
            else:
                neighbor_classes[train_set[distance[val_set[i]][j][0]]] = 1/(j+1)
                         
            if checkinhyperparam(j+1,k_neighbors):
                predicted_class = sorted(neighbor_classes.items(), key=lambda kv: kv[1], reverse=True)
                valclass[val_set[i]].append(predicted_class[0][0])
    
    df_fit = pd.DataFrame.from_dict(valclass, orient='index')
    GTlabels = getlabels(val_set_dict)
    df_fit['GT'] = GTlabels
    accu = []
    for column in df_fit:
        if column != 'GT':
            accu.append(accuracy_metric(df_fit['GT'].to_numpy(), df_fit[column].to_numpy()))

    return accu



def KNN_predict(df, X_train, X_test, k):
    distance = {}
    y_test = []
    
    for i in tqdm(range(len(X_test))):
        distance[X_test[i]]=[]
        for key in X_train:
            score = cosine_sim(df, key, X_test[i])
            distance[X_test[i]].append((key,score))
            
        distance[X_test[i]].sort(reverse=True, key=lambda kv: kv[1])
        
        neighbor_classes = {}

        # Used harmonic series weighing scheme
        for j in range(k):
            if X_train[distance[X_test[i]][j][0]] in neighbor_classes:
                neighbor_classes[X_train[distance[X_test[i]][j][0]]] += 1/(j+1)
            else:
                neighbor_classes[X_train[distance[X_test[i]][j][0]]] = 1/(j+1)
        
        predicted_class = sorted(neighbor_classes.items(), key=lambda kv: kv[1], reverse=True)        
        y_test.append(X_test[i]+' '+predicted_class[0][0])
        
        
    return y_test

if __name__ == '__main__':

    train_input = input("training file name: ")
    test_input = input("testing file name: ")
    # train_input = "./corpus1_train.labels"
    # test_input = "./corpus1_test.list"

    vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words='english', min_df=1)
    
    training_dict, X_test = preprocess(train_input, test_input)
    cor1, paths1,[len1, len2] = getallcorpus(training_dict, X_test)
    tfidf1 = vectorizer.fit_transform(cor1)
    df = pd.DataFrame(tfidf1.toarray(),columns=vectorizer.get_feature_names(), index=paths1)


    tuning_df = pd.DataFrame.from_dict(training_dict, orient='index')

    # Do a 5-fold cross-validation
    print("Start 5-fold cross validation for hyper parameter tuning. \n")
    kf = KFold(n_splits = 5)
    train_set_M = []
    val_set_M = []
    for train_index, test_index in kf.split(tuning_df):
        train_set_M.append(tuning_df.iloc[train_index].to_dict()[0])
        val_set_M.append(tuning_df.iloc[test_index].to_dict()[0])
    acculist = []
    for i in tqdm(range(5)):
        accu_list = KNN_fit(df, train_set_M[i], val_set_M[i], k_neighbors)
        acculist.append(accu_list)

    aveaccu = np.mean(acculist, axis=0)

    # Get the best k value
    best_k = k_neighbors[np.argmax(aveaccu)]
    print("\nThe best k value is: ")
    print(best_k)
    print("\n")
    
    # Use the best k value to do KNN prediction
    print("Getting predictions. \n")
    y_predict = KNN_predict(df, training_dict, X_test, best_k)
    
    
    filename = input("output file name: ")
    outfile = open(filename, 'w')
    for i in y_predict:
        outfile.write(i+'\n')
    
    outfile.close()