import numpy as np
import cmath
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
from tqdm import tqdm
from glob import glob
import os 
from random import choice
import pandas as pd
from collections import defaultdict

data_path ="./NILM_datasets/SmartMeter"

class SmartMeter:
    def __init__(self):

        self.data = glob(data_path + "/*/*.*")

        self.labels_dict = {'iphone13':0, 'computer':1, 
                            'clothes-dryer':2,
                            #'washing-machine':3, 
                            'electric-scooter':4,
                            'hair-dryer':5}

    def save_dataset(self):

        self.voltage = []
        self.current = []
        self.labels = []

        for sample in self.data:
            #print(sample)
            df = pd.read_csv(sample)
            v,i = df['V'].to_numpy(), df['I'].to_numpy()

            label = self.labels_dict[sample.split("/")[3]]

            voltages = np.array_split(v, 50)
            currents = np.array_split(i, 50)

            for volt, curr in zip(voltages, currents):

                self.voltage.append(volt)
                self.current.append(curr)
                self.labels.append(label)

        with open(data_path + "/dataset.npy", 'wb') as f:
            pickle.dump([np.asarray(self.voltage, dtype='float32'), np.asarray(self.current, dtype='float32'), self.labels], f)
            f.close()

    def load_dataset(self):

        with open(data_path + "/dataset.npy", 'rb') as f:
            self.dataset = pickle.load(f, encoding='latin1')
            f.close

            return self.dataset

    def plot_dataset(self, mode='all'):

        self.load_dataset()
        d = defaultdict(list)

        voltage, current, labels = self.dataset

        for v,i,l in zip(voltage, current, labels):
            d[l].append((v,i))

        if mode == 'all':
            idx = 0
            for key,value in self.labels_dict.items():
                plt.subplot(2,3,idx + 1)
                v,i = choice(d[value])
                print(v.shape, i.shape)
                plt.title(key)
                plt.plot(v, i)
                idx+=1
            plt.show()

    def plot_comparison(self):
        self.load_dataset()
        d = defaultdict(list)

        voltage, current, labels = self.dataset

        for v,i,l in zip(voltage, current, labels):
            d[l].append((v,i))

        #washing_machine = choice(d[3])
        #

        #plt.plot(washing_machine[1])
        #plt.show()

        appliances = []

        for key,value in self.labels_dict.items():
            v,i = choice(d[value])
            appliances.append([key , v, i])

        idx = 0
        for app in appliances:
            plt.subplot(2,3,idx+1)
            plt.title(app[0], fontsize=22)
            plt.xticks([np.min(app[1]), np.max(app[1])],fontsize=18)
            plt.yticks([np.min(app[2]), np.max(app[2])],fontsize=18)
            plt.xlabel('v', fontsize=20)
            plt.ylabel('i', fontsize=20)
            plt.plot(app[1], app[2])
            idx+=1
        #plt.savefig("./original.pdf")
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)
        plt.show()

        idx = 0
        for app in appliances:
            plt.subplot(2,3,idx+1)
            plt.title(app[0], fontsize=22)
            mask = [bool(d % 16 == 0) for d in range(0, app[1].shape[0])]
            v = app[1][mask]
            i = app[2][mask]
            plt.xticks([np.min(v), np.max(v)],fontsize=18)
            plt.yticks([np.min(i), np.max(i)],fontsize=18)
            plt.xlabel('v', fontsize=20)
            plt.ylabel('i', fontsize=20)
            plt.xlabel('v', fontsize=18)
            plt.ylabel('i', fontsize=18)
            plt.plot(v,i, 'o', color='black')
            idx+=1
        #plt.savefig("./reduced.pdf")
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)
        plt.show()
            


if __name__ == "__main__":
    sm = SmartMeter()
    sm.save_dataset()
    sm.plot_comparison()





