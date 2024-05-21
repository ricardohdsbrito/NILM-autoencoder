import matplotlib.pyplot as plt
import numpy as np
from NILM_dataset import *
from trainer import *
import random 
import sys
from smartmeter import SmartMeter
from dataset_utils import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

INPUT_SHAPE = 275

def visualize(data, feature=4, name=''):

    print("visualizing with %s features" % str(feature))
    

    trainer = Trainer(data, batch_size = 512,
                       learning_rate = 1e-2,
                       epochs =100,
                       input_shape = INPUT_SHAPE,
                       num_features=feature,
                       shuffle=True)

    trainer.load("models/%s.model_%s" % (name, str(feature)))
    trainer.visualize()


def pretrain_models(data,input_shape, feature=4, name='', epochs=500):

    print("training with %s features" % str(feature))
    

    trainer = Trainer(data, batch_size = 512,
                       learning_rate = 1e-2,
                       epochs =epochs,
                       input_shape = input_shape,
                       num_features=feature,
                       shuffle=True)

    trainer.train()

    trainer.save("models/%s_shape_%s.features_%s" % (name, str(input_shape),str(feature)))

def run_ml_classifier(features, labels, clf = None):
    
    training_data = []
    testing_data = []
    data = {}

    for idx, label in enumerate(labels):
        if label not in data:
            data[label] = []
            data[label].append(features[idx])
        else:
            data[label].append(features[idx])

    for key, value in data.items():
        total = len(value)
        tr = int(0.9 * total)
        training_data.append(value[:tr])
        testing_data.append(value[tr:])
        
    training_data = [lst for sublst in training_data for lst in sublst]
    testing_data = [lst for sublst in testing_data for lst in sublst]

    random.shuffle(training_data)
    random.shuffle(testing_data)

    end = -1
        
    training_data = np.asarray(training_data, dtype='float32')
    testing_data = np.asarray(testing_data, dtype='float32')

    training_labels = np.asarray(training_data[:,end], dtype='int32')
    testing_labels = np.asarray(testing_data[:,end], dtype='int32')

    training_data = np.asarray(training_data[:, 0:end], dtype='float32')
    testing_data = np.asarray(testing_data[:, 0:end], dtype='float32')

    print(testing_data.shape)

    clf.fit(training_data, training_labels)

    accuracy = float(np.sum(clf.predict(testing_data) == testing_labels))/testing_labels.shape[0]

    print("acc: %s" % str(accuracy))

    return accuracy



def run_individual_experiments(data, labels, reduction_factor=2, feature=4, name='', clf=None):

    """
    This function extracts features from a model trained
    by the train_feature_extractors function and uses th
    ese features in load classification using either KNN
    XGBoost or Random Forest
    """

    print("Testing with %s features" % str(feature))

    mask = [bool(d % reduction_factor == 0) for d in range(0, data.shape[1])]

    input_shape = np.sum(mask)

    data = NILMDataset(data[:, mask])

    trainer = Trainer(data, batch_size = 512,
                       learning_rate = 1e-2,
                       epochs =100,
                       input_shape = input_shape,
                       num_features=feature,
                       shuffle=False)

    trainer.load("models/%s_shape_%s.features_%s" % (name, str(input_shape),str(feature)))

    features = trainer.extract_features()
    features = np.concatenate(features, axis=0)
    targets = np.expand_dims(np.asarray(labels), axis=1)

    features = np.concatenate([features,targets], axis=1)

    return run_ml_classifier(features, labels, clf=clf)

   
 
def run_joint_experiments(data, clf=None):

    """
    This function extracts features from different models
    (trained by the train_feature_extractor function) and
    combines them into a single feature vector, e.g: it i
    s possible to extract features from a model trained o
    n instantaneous p and another model trained on instan
    taneous i, then these features are combined here into
    a single feature vector for load classification, as d
    escribed in the paper.
    """
    
    labels = data["labels"]
    combined_features = []

    for name, dataset, feature, reduction_factor in zip(data["names"], data["datasets"], data["features"], data["reduction_factors"]):

        mask = [bool(d % reduction_factor == 0) for d in range(0, dataset.shape[1])]

        input_shape = np.sum(mask)

        dataset = NILMDataset(dataset[:, mask])

        trainer = Trainer(dataset, batch_size = 512,
               learning_rate = 1e-2,
               epochs =100,
               input_shape = input_shape,
               num_features=feature,
               shuffle=False)

        trainer.load("models/%s_shape_%s.features_%s" % (name, str(input_shape),str(feature)))

        features = trainer.extract_features()
        features = np.concatenate(features, axis=0)
        combined_features.append(features)

    targets = np.expand_dims(np.asarray(labels), axis=1)
    combined_features = np.concatenate(combined_features, axis=1)
    combined_features = np.concatenate([combined_features,targets], axis=1)

    run_ml_classifier(combined_features, labels, clf=clf)


def train_feature_extractors(dataset, model_name='', sizes = [2, 4, 8, 16, 32, 64], features = [2, 4, 8, 16, 32, 64, 128]): 

    """
    This function trains models to extract features
    from different input sampling frequencies.
    Frequencies are reduced by powers of 2, as spec
    ified in the sizes array. The features are incr
    eased by powers of 2 as specified in the featur
    es arrat.
    """

    print(dataset.shape)

    for size in sizes:

        mask = [bool(d % size == 0) for d in range(0, dataset.shape[1])]

        input_shape = np.sum(mask)

        name = model_name
        print("Training: ", name)
        print("Reduction factor: ", size)
        print("Reduction factor: ", size)

        data = NILMDataset(dataset[:, mask])

        for feature in features:
            pretrain_models(data, input_shape, feature=feature, name=name, epochs=100)

if __name__ == "__main__":

    #1. Load the dataset

    #Here we load our own dataset from the NILM_datasets/SmartMeter folder

    sm = SmartMeter()
    v, i, labels = sm.load_dataset()

    #We can also load v,i and p,q from the REDD dataset, depending on what data we want to use
    vi, pq, labels = load_datasets(house="house_3", num_classes=22)

    #2. Train the models

    train_feature_extractors(vi.i, model_name='REDD-i')

    #3. Load NILM classifiers

    xgb = RandomForestClassifier(n_estimators=100,
                                     max_depth=10)
     
    knn = KNeighborsClassifier(n_neighbors=10)

    rf = RandomForestClassifier(n_estimators=100,
                                      max_depth=10)
    classifiers = [xgb, knn, rf]

    #4-a. Run the load classification with sampling reduction and feature extraction
    # Extracting features from a single model (here we used our own data)

    run_individual_experiments(vi.i, labels, reduction_factor=16, feature=4, name='REDD-i', clf=rf)

    #4-b. Run the load classification with sampling reduction and extract features
    # from multiple models (here we used the REDD dataset, combining v-i and p-q features.

    data_joint = {"datasets":[pq.p_inst, vi.i],
                  "names":["REDD-p", "REDD-i"],
                  "features":[4, 4],
                  "reduction_factors": [16,16],
                  "labels":labels}

    run_joint_experiments(data_joint, clf=rf)

    #visualize(data, feature=8, name='house5')
   
