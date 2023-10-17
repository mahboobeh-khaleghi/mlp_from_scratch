import numpy as np
import math
import pickle
import dataloaders

class Cifar10DataLoader:
    def __init__(self, pickles_path, batch_size, preprocessing_method):
        self.data = list()
        self.labels = list()
        self.batch_size = batch_size
        self.read_batches(pickles_path)
        self.num_calsses = int(self.labels.max() - self.labels.min() + 1)
        
        # Preprocessing Method
        assert preprocessing_method.lower() in ["normalized", "standardized", "no"], "Error!! Undefined preprocessing. Preprocessing method must be one of this elements: ['normalized', 'standardized', 'no']"
        self.preprocessing_method = preprocessing_method
        
        # Preprocessing Data
        if self.preprocessing_method == "normalized":
            # minimum of data
            self.min = np.min(self.data)
            
            # maximum of data 
            self.max = np.max(self.data)
            
            # normalization
            self.data = (self.data - self.min) / (self.max - self.min)
            
        elif self.preprocessing_method == "standardized":
            # mean of data
            self.mean = np.mean(self.data)
            
            # std of data
            self.std = np.std(self.data)
            
            #standardized
            self.data = (self.data - self.mean) / self.std
        else:
            pass
        
    def read_pickle(self, path):
        with open (path, "rb") as file:
            pickle_dict = pickle.load(file , encoding = "latin1")
        return np.array(pickle_dict["labels"]), pickle_dict["data"]
    
    def read_batches(self, pickles_path):
        
        for path in pickles_path:
            path_label, path_data = self.read_pickle(path)
            self.data.append(path_data)
            self.labels.append(path_label)
            
        self.data = np.concatenate(self.data).astype(np.float64)
        self.labels = np.concatenate(self.labels).astype(int)
        
    def get_batch(self,ix):
        start_ix = ix * self.batch_size
        end_ix = (ix+1)* self.batch_size
        if end_ix > self.data.shape[0]:
            end_ix = self.data.shape[0]
        
        batch_data = self.data[start_ix :end_ix , :]
        batch_labels = self.labels [ start_ix :end_ix ]
        
        one_hot_batch_labels = np.zeros((self.batch_size, self.num_calsses))
        one_hot_batch_labels[np.arange(self.batch_size), batch_labels.astype(int)] = 1
        
        return batch_data , one_hot_batch_labels
    
    def get_num_batches(self):
        num_data = self.data.shape[0]
        return math.ceil(num_data / self.batch_size)
    
    def __len__(self):
        return self.data.shape[0]
        
    def read_batches(self, pickles_path):
        
        for path in pickles_path:
            path_label, path_data = self.read_pickle(path)
            self.data.append(path_data)
            self.labels.append(path_label)
            
        self.data = np.concatenate(self.data).astype(np.float64)
        self.labels = np.concatenate(self.labels).astype(np.float64)
        
            
    def get_item(self, ix):
        return self.data[ix,:],self.labels[ix]