import numpy as np
import cmath
from k_means_centroid import *
from vi import *
from pq import *

def preprocess_data(pq, labels, threshold=8):

    #The reson we use a threshold here is because the pq.areas were calculated using the GPU and the batch size was a multiple of 10, so we need to remove 8 elements in each feature to match the number in pq.areas

    labels = labels[:-threshold]

    fts = [pq.areas, pq.segs[:-threshold], pq.lengths[:-threshold], pq.means[:-threshold], pq.p_avg[:-threshold], pq.q_avg[:-threshold], pq.slopes[:-threshold]]
    nrm_fts = list(map(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), fts))

    return (nrm_fts, labels)
 

def shift_signal(signal):
    
    signalFFT = np.fft.rfft(signal)
    ## Get Phase
    signalPhase = np.angle(signalFFT)
    
    ## Phase Shift the signal +90 degrees
    newSignalFFT = signalFFT * cmath.rect( 1., -np.pi/2 )
    ## Reverse Fourier transform
    newSignal = np.fft.irfft(newSignalFFT)
    newSignal = np.append(newSignal, [newSignal[0]])
    return np.asarray(newSignal, dtype='float32')



def load_datasets(house="house_5", num_classes=26):

    vi = VI(house)
    
    print("Reading data...")
    
    vi.read_data() # iv.i, iv.v
    voltage = np.expand_dims(vi.v[0], axis=0)
    voltage = np.repeat(voltage, vi.i.shape[0], axis=0)
    vi.v = voltage

    print("done!")

    kmc = KMC(num_classes,275,1000)
    
    kmc.load(house)
    pq = PQ(house)

    print("loading (PQ)...")
    pq.load()
    print("done!")

    return vi, pq, kmc.labels

