import numpy as np

class VI():

    def __init__(self, house, base = './NILM_datasets/', datadir="REDD/"):
        
        self.current_train = base + datadir + house + "/current_1.dat"
        self.current_train2 = base + datadir + house + "/current_2.dat"
        self.voltage_file = base + datadir + house + "/voltage.dat"

    
    
    def read_data(self):

        current_data = list(open(self.current_train, "r"))
        current_data2 = list(open(self.current_train2, "r"))
        
        voltage_data = list(open(self.voltage_file, "r"))
        
        current_data = [row.strip("\n").split(" ")[2:] for row in current_data]
        
        
        current_data2 = [row.strip("\n").split(" ")[2:] for row in current_data2]
        
        
        current_data.extend(current_data2)

        current_data = np.asarray(current_data, dtype='float32')
        
        voltage_data = np.asarray([row.strip("\n").split(" ")[2:] for row in voltage_data], dtype='float32')

        self.i = current_data 
        self.v = voltage_data
        
