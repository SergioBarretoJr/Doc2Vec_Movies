import logging
import random
import numpy as np
import pandas as pd
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter="\t")
    x_train, x_test, y_train, y_test = train_test_split(dataset.review, dataset.sentiment, random_state=0, test_size=0.2)
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
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors

def train_doc2vec(corpus):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=5,  #Ignores all words with total frequency lower than this
                          window=5,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=1)  # dm=1 means 'distributed memory' (PV-DM:predict a center word from the randomly
                                                                        # sampled set of words by taking as input — 
                                                                        # the context words and a paragraph id.)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW: ignores the context words in
                                                                                      # the input, but force the model
                                                                                      # to predict words randomly sampled
                                                                                      # from the paragraph in the output.
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    for epoch in range(5):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002


    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    return d2v

def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), 300, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model

def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), 300, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))

def predict_Class(d2v, classifier, strings):
    logging.info("Classifing new instance")
    vec = doc2vec.infer_vector(doc_words=strings.split(), alpha=0.025, steps=20)
    if (classifier.predict(vec.reshape(1, -1))==1):
        print("Avaliação positiva\n")
    else: print("Avaliação negativa\n")


def view_Embeddings(d2v_model,doc,path_vec,LearnR,epoch):
    test_docs = doc
    test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]
    output = open(path_vec, "w")
    vec_docs=[]
    for d in test_docs:
        output.write(" ".join([str(x) for x in d2v_model.infer_vector(d, alpha=LearnR, steps=epoch)]) + "\n")
        vec_docs.append(d2v_model.infer_vector(d, alpha=LearnR, steps=epoch))
    output.flush()
    output.close()
    return vec_docs


if __name__ == "__main__":

    #Separando treinamento, teste, classe
    x_train, x_test, y_train, y_test, all_data = read_dataset('dataset.csv')

    #construindo a representaçao embedding
    #doc2vec = train_doc2vec(all_data)

    #carregando modelo criado
    doc2vec = Doc2Vec.load("d2v.model")

    #salvar representação vetorial da base
    #text_vec=view_Embeddings(doc2vec,'dataset.csv','dataset_vectors.txt', 0.025, 3)
    #print(text_vec)

    #usando a representação vetorial para treinar classificador
    classifier = train_classifier(doc2vec, x_train, y_train)

    #testando classificador gerado
    test_classifier(doc2vec, classifier, x_test, y_test)


    # testando nova avaliação de filme
    strings = 'Avengers is a very good film. i loved it so much'
    string2 = 'Avengers is the worst film. i hated it'
    predict_Class(doc2vec, classifier, strings)
    predict_Class(doc2vec, classifier, string2)

