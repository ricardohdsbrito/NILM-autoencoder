import numpy as np
from utils import *
import cmath
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
from tqdm import tqdm
import os 

class PQ:

    def __init__(self, house, base = './NILM_datasets/', datadir="REDD/", save_path=''):
        if not os.path.isdir(base + datadir + house + save_path):
            os.mkdir(base + datadir + house + save_path)

        self.p_inst_path = base + datadir + house + save_path + '/p_inst.npy'
        self.q_inst_path = base + datadir + house + save_path +'/q_inst.npy'

        self.p_avg_path = base + datadir + house + save_path +'/p_avg.npy'
        self.q_avg_path = base + datadir + house + save_path +'/q_avg.npy'

        self.areas_path = base + datadir + house + save_path +'/pq_areas.npy'

        self.means_path = base + datadir + house + save_path +'/pq_mean_curve.npy'
        self.lengths_path = base + datadir + house + save_path +'/pq_cur_len.npy'
        self.segs_path = base + datadir + house + save_path +'/pq_segs.npy'
        self.slopes_path = base + datadir + house + save_path +'/pq_slopes.npy'
        

    def pq_inst(self, vol, cur):

        self.voltage_data = vol
        self.current_data = cur
        
        vltg_shifted = self.shift_signal(self.voltage_data)
        self.p_inst = self.current_data * self.voltage_data
        self.q_inst = self.current_data * vltg_shifted

    def pq_avg(self, vol, cur):

        self.voltage_data = vol
        self.current_data = cur
        
        vltg_shifted = self.shift_signal(self.voltage_data)
        
        self.p_avg = self.current_data * self.voltage_data
        self.q_avg = self.current_data * vltg_shifted
    
        self.p_avg = np.mean(self.p_avg, axis = 1)
        #self.p_avg = ((120 / np.sqrt(np.mean(self.voltage_data ** 2, axis = 1))) ** 2)*self.p_avg
        self.q_avg = np.mean(self.q_avg, axis = 1)
    

    def shift_signal(self, signal):
        
        signalFFT = np.fft.rfft(signal)
        ## Get Phase
        signalPhase = np.angle(signalFFT)
        
        ## Phase Shift the signal +90 degrees
        newSignalFFT = signalFFT * cmath.rect( 1., -np.pi/2 )
        ## Reverse Fourier transform
        newSignal = np.fft.irfft(newSignalFFT)
        newSignal = np.append(newSignal, np.expand_dims(newSignal[:,0], axis=1), axis=1)
        return np.asarray(newSignal, dtype='float32')

    def get_features(self, voltage, current):

        print("calculating intant pq")
    
        self.pq_inst(voltage, current)

        print("calculating average pq")
        
        self.pq_avg(voltage, current)

        print("calculating pq area")

        self.get_area(10, 10000)

        print("calculating line seg")

        self.get_line_segment()

        print("calculating mean curve")

        self.get_mean_curve()

        print("calculating curve length")

        self.get_curve_length()

        self.save()

    def get_curve_length(self):
        
        self.lengths = []

        for i in tqdm(range(self.p_inst.shape[0])):
            length = 0
            for j in range(self.p_inst[i].shape[0] - 1):
                length += (self.p_inst[i][j] - self.p_inst[i][j+1])**2 +\
                          (self.q_inst[i][j] - self.q_inst[i][j+1])**2
            
            self.lengths.append(length)

        self.lengths = np.asarray(self.lengths, dtype='float32')
        
    def get_mean_curve(self):

        self.means = []
        self.slopes = []
        for i in tqdm(range(self.p_inst.shape[0])):
            self.means.append(np.mean((self.p_inst[i]+self.q_inst[i])/2, axis=0))

            #plt.plot(self.p_inst[i], self.q_inst[i])
            #plt.plot(self.p_inst[i],  self.q_inst[i]/2)
            middle = int(self.p_inst.shape[1]/2)
            meanx = [np.mean(self.p_inst[i][0:middle]), np.mean(self.p_inst[i][middle:])]
            meany = [np.mean(self.q_inst[i][0:middle]), np.mean(self.q_inst[i][middle:])]
            #print meanx
            #print meany
            #plt.plot(meanx, meany)
            #plt.show()
            self.slopes.append((meany[1]-meany[0])/(meanx[1]-meanx[0]))
        self.means = np.asarray(self.means, dtype='float32')
            
    def get_line_segment(self):

        self.segs = []
        self.angs = []
        self.slopes = []
        
        for i in tqdm(range(self.p_inst.shape[0])):
            #print i
            pi = np.expand_dims(self.p_inst[i], axis=0)
            pi = np.repeat(pi, pi.shape[0], axis=0)
            #pi = np.transpose(pi)
        
            pj = np.expand_dims(self.p_inst[i],1)

            qi = np.expand_dims(self.q_inst[i], axis=0)
            qi = np.repeat(qi, qi.shape[0], axis=0)
            #qi = np.transpose(qi)
            
            qj = np.expand_dims(self.q_inst[i],1)

            diff_p = (pj-pi)**2
            diff_q = (qj-qi)**2
            res = diff_p + diff_q
            hip = np.max(np.sqrt(diff_p + diff_q))
            self.segs.append(hip)
        
            idx = np.unravel_index(np.argmax(res, axis=None), res.shape)
        
        
            pointx = [self.p_inst[i, idx[0]], self.p_inst[i, idx[1]]]
            pointy = [self.q_inst[i, idx[0]], self.q_inst[i, idx[1]]]

            c1 = abs(pointx[0] - pointx[1])
            c2 = abs(pointy[0] - pointy[1])
            sinc = np.arcsin(c2 / hip)
            self.angs.append(sinc)
            self.slopes.append((pointy[1]-pointy[0])/(pointx[1]-pointx[0]))
            #print pointx, pointy
            '''
            plt.title("Longest line segment that fits in the p-q curve")
            plt.xlabel("p")
            plt.ylabel("q")
            plt.plot(self.p_inst[i], self.q_inst[i])
            plt.plot(pointx, pointy)
            plt.savefig("plots/%s.pdf" % str(i))
            plt.close()
            '''
        self.segs = np.asarray(self.segs, dtype='float32')
            
    def get_area(self, batch_size, max_points):
        
        font = {'family' : 'normal',
                'size'   : 16}

        matplotlib.rc('font', **font)
        
        total = self.p_inst.shape[0] - 8
        self.areas = []
        
        for batch in tqdm(range(0, total, batch_size)):
            p = self.p_inst[batch:batch+batch_size]
            q = self.q_inst[batch:batch+batch_size]

            p_min = np.min(p, axis=1)
            p_max = np.max(p, axis=1)
            q_min = np.min(q, axis=1)
            q_max = np.max(q, axis=1)

            p_span = p_max - p_min
            q_span = q_max - q_min

            p_rand = np.expand_dims(p_min, axis=1) + (np.random.rand(10, max_points) * np.expand_dims(p_span, axis=1))
            q_rand = np.expand_dims(q_min, axis=1) + (np.random.rand(10, max_points) * np.expand_dims(q_span, axis=1))

            pts = np.ravel(np.dstack((p_rand,q_rand))).astype('float32')
            
            pol_x = 550 # 275 * 2
            pol_y = batch_size
            k = max_points * 2
            result = np.ravel(np.zeros((batch_size, max_points))).astype('float32')
        
            polygon = np.ravel(np.dstack((p,q))).astype('float32')
            
            IsInsidePolygon(polygon, pol_x, pol_y, pts, k, result)

            result = np.reshape(result, (batch_size, max_points))

            rect_area = p_span * q_span
            poly_area = (rect_area * np.sum(result, axis=1))/max_points
            self.areas.append(poly_area)
        
            """  
            for i in range(batch_size):
            
                #print "the area is %f" % poly_area[i]
                
                res = result[i]

                pol = list(group(polygon[i*550:550*i + 550],2))
                pts2 = list(group(pts[i*k:k*i+k],2))
                pts3 = list(group(pts[i*k:k*i+k],2))
        
                pts2 = [pts2[j] for j in range(max_points) if res[j] == 1]


                #plt.sublot(1,2,1)
                #plt.plot(*zip(*pts2), marker='o', linestyle='', ms=1, color='red')
                #plt.plot(p[i], q[i])
                #plt.subplot(1,2,2)
                
                #plt.savefig('plots/in_%s.pdf' % str(i+(batch*batch_size)))
                #plt.close()

                plt.figure(figsize=(10,6))
                plt.subplot(1,2,1)
                plt.title("Total random points inside rectangle")
                plt.plot(*zip(*pts3), marker='o', linestyle='', ms=1, color='blue')
                plt.plot(p[i], q[i], linewidth=3.5)
                plt.xlabel("p")
                plt.ylabel("q")
                
                plt.subplot(1,2,2)
                plt.title("Random points inside p-q graph")
                plt.plot(*zip(*pts2), marker='o', linestyle='', ms=1, color='blue')
                plt.plot(p[i], q[i], linewidth=3.5)
                plt.xlabel("p")
                plt.ylabel("q")
                #plt.savefig('plots/%s.pdf' % str(i+(batch*batch_size)))
                plt.tight_layout()
                plt.show()
            """

        self.areas = np.ravel(np.asarray(self.areas))
        
    def plot_pq(self, labels):
        
        
        df = pd.DataFrame(dict(x=self.p_avg, y=self.q_avg, label=labels))
        groups = df.groupby('label')

    
        label_colors = colors(len(groups))
        label_colors = ['#%02x%02x%02x' % (color) for color in label_colors]
        labels = range(len(groups))
    
        fig, ax = plt.subplots()
        ax.margins(0.05) 
        ax.set_title("PQ for different appliances")
        plt.xlabel("Active Power (P)")
        plt.ylabel("Rective Power (Q)")

        plotter = [group for (name, group) in groups]
        plotter = [(group, l) for (group, l) in zip(plotter, labels)]

        for group, col in zip(plotter, label_colors):
            ax.plot(group[0].x, group[0].y, marker='o', linestyle='', ms=5, color=col, label = group[1])
        ax.legend()

        plt.show()#savefig("paper_outline/figures/pq_avg.pdf")


    def save(self):
        
        with open(self.p_inst_path, 'wb') as f:
            pickle.dump(self.p_inst, f)
            f.close()
        with open(self.q_inst_path, 'wb') as f:
            pickle.dump(self.q_inst, f)
            f.close()
        with open(self.p_avg_path, 'wb') as f:
            pickle.dump(self.p_avg, f)
            f.close()
        with open(self.q_avg_path, 'wb') as f:
            pickle.dump(self.q_avg, f)
            f.close()
        with open(self.areas_path, 'wb') as f:
            pickle.dump(self.areas, f)
            f.close()
        with open(self.means_path, 'wb') as f:
            pickle.dump(self.means, f)
            f.close()
        with open(self.lengths_path, 'wb') as f:
            pickle.dump(self.lengths, f)
            f.close()
        with open(self.segs_path, 'wb') as f:
            pickle.dump(self.segs, f)
            f.close()
        with open(self.slopes_path, 'wb') as f:
            pickle.dump(self.slopes, f)
            f.close()
        
        
        
            
    def load(self):
        with open(self.p_inst_path, 'rb') as f:
            self.p_inst = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.q_inst_path, 'rb') as f:
            self.q_inst = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.p_avg_path, 'rb') as f:
            self.p_avg = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.q_avg_path, 'rb') as f:
            self.q_avg = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.areas_path, 'rb') as f:
            self.areas = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.means_path, 'rb') as f:
            self.means = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.lengths_path, 'rb') as f:
            self.lengths = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.segs_path, 'rb') as f:
            self.segs = pickle.load(f, encoding='latin1')
            f.close()
        with open(self.slopes_path, 'rb') as f:
            self.slopes = pickle.load(f, encoding='latin1')
            f.close()
        
        

    
