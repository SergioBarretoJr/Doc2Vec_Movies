import logging
import random
import numpy as np
import pandas as pd
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score

import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_dataset(path_train,path_test):
    dataset_trains = pd.read_csv(path_train, header=0, delimiter=";")
    dataset_test = pd.read_csv(path_test, header=0, delimiter=";")
    dataset_trains=pd.DataFrame(dataset_trains,columns=['0','1'])
    dataset_test = pd.DataFrame(dataset_test, columns=['0', '1'])

    x_train=dataset_trains['0']
    y_train = dataset_trains['1']
    x_test = dataset_test['0']
    y_test = dataset_test['1']
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data

def label_sentences(corpus, label_type):

    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled

def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors

def train_doc2vec(corpus,w,e,Vs):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  #Ignores all words with total frequency lower than this
                          window=w,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=Vs,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.01,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dbow_words=0,
                          hs=0,
                          seed=123,
                          dm=0,)  # dm=1 means 'distributed memory' (PV-DM:predict a center word from the randomly
                                                                        # sampled set of words by taking as input — 
                                                                        # the context words and a paragraph id.)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW: ignores the context words in
                                                                                      # the input, but force the model
                                                                                      # to predict words randomly sampled
                                                                                      # from the paragraph in the output.
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    for epoch in range(e):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002


    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    return d2v

def train_classifier(d2v, training_vectors, training_labels,Vs):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), Vs, 'Train')
    model = KNeighborsClassifier()
    model.fit(train_vectors, np.array(training_labels))
    return model

def test_classifier(d2v, classifier, testing_vectors, testing_labels,Vs,cv):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), Vs, 'Test')
    scores=cross_val_score(classifier, test_vectors, testing_labels, cv=cv)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def build_PVDBOW(input_trains1,input_test1, w1,e1,cv,plot_Class=False):
    input_trains=input_trains1
    input_test = input_test1
    dataset = pd.read_csv(input_trains, header=0, delimiter=";")
    dataset_test= pd.read_csv(input_test, header=0, delimiter=";")
    dataset = pd.DataFrame(dataset, columns=['0', '1'])
    dataset_test=pd.DataFrame(dataset_test, columns=['0', '1'])
    if (plot_Class):
        print("Descrição da Base:")
        print("%Positivo: ",np.mean(dataset['1']))
        print("%Positivo validação: ",np.mean(dataset_test['1']))
        sns.catplot(x="1", kind="count", data=dataset,color='blue')
        sns.catplot(x="1", kind="count", data=dataset_test, color='black')
        plt.show()

    #Separando treinamento, teste, classe
    x_train, x_test, y_train, y_test, all_data = read_dataset(input_trains,input_test)
    print("# de tweets: ",len(x_train)+len(x_test))
    print("# de tweets treino: ",len(x_train))
    print("# de tweets test: ", len(x_test))
    print("% de tweets test: ", len(x_test)*100/(len(x_train)+len(x_test)))
    w=w1
    e=e1
    #construindo a representaçao embedding
    doc2vec = train_doc2vec(all_data,w,e)

    #usando a representação vetorial para treinar classificador
    classifier = train_classifier(doc2vec, x_train, y_train)

    #testando classificador gerado
    test_classifier(doc2vec, classifier, x_test, y_test,cv)

