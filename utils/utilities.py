import yaml
import argparse
import numpy as np

class Struct:
    def __init__(self, entries): 
        self.__dict__.update(entries)
        
def dict2namesspace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namesspace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
        
def read_config(config_path):
    with open (config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    return dict2namesspace(config_dict)

def accuracy(output, one_hot_labels):
    """
        output: output of the model
            * shape: (batch_size, num_labels)
        labels: correct label of each output
            * shape: (batch_size)
            
        return:
            * accuracy of classification
    """
    # import pdb; pdb.set_trace()
    # obtaining predicted labels
    predicted_labels = np.argmax(output, axis=1)    # shape: (batch_size,)
    
    # comparison between predicted and correct labels and obtaining correct classified instances
    labels = np.argmax(one_hot_labels, axis=1)
    correct_classified = predicted_labels == labels
    
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # accuracy = (correct_classified) / batch_size 
    accuracy = np.sum(correct_classified) / output.shape[0]
    
    return accuracy


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(
        self,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.count = start_count
        self.avg = start_avg
        self.sum = start_sum

    def reset(self):
        """
            Initialize 'sum', 'count', and 'avg' with 0.
        """
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, values):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.sum += sum(values)
        self.count += len(values)
        self.avg = self.sum / self.count

def get_args():
    parser = argparse.ArgumentParser(description="Config Path Argument Parser")
    
    # Config path
    parser.add_argument("-c", "--config", dest="config_path", type=str , help="config path")
    
    options = parser.parse_args()
    
    return options
    
    