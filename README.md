# Down Sampling Neural Network Autoencoder Feature Extraction for Non-Intrusive Load Monitoring

This repository contains the code implementation of the algorithm proposed in our paper "Down Sampling Neural Network Autoencoder Feature Extraction for Non-Intrusive Load Monitoring"

## Instalation

Download the ```NILM_datasets``` folder from https://drive.google.com/drive/folders/1p-XwELAN1NyjW7xwE2Ox5EEba87HbIX0?usp=sharing and place it in the root of this project.

```bash
pip3 install -r requirements.txt
```

## Usage

Simply run the following command:

```bash
python3 NILM_autoencoder.py
```

The file NILM_autoencoder.py contains contains comments that explain what each function does:

```python
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
```

The ```train_feature_extractors``` function trains the models that perform feature extraction.

The datasets are stored in the NILM_datasets folder, which contains the REDD dataset inside the ```REDD``` folder and our own smart meter data located in the SmartMeter folder. The SmartMeter folder contains data that we collected for 5 devices as specified in our paper.

The model folder is where the trained models are saved and it already contains some pre-trained models for the REDD dataset, using the instantaneous current and instantaneous power.

The experiments can be performed individually (run_individual_experiments) where the function takes a dataset (vi.i is the instantaneous current (i) dataset loaded from the REDD dataset), a ```reduction_factor``` which specifies the sampling reduction factor of the selected model,  along with the ```feature``` parameter, which specifies the number of features extracted by the selected model and the ```name``` parameter, which specifies the name of the selected model.

The experiements can be performed jointly in which features extracted from multiple models can be combined into a single feature vector. This is done by calling the ```run_joint_experiments``` function with the data passed in a dictionary as shown in the ```data_joint``` dictionary in the example above.


