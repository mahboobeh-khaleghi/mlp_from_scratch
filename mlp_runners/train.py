import numpy as np
import pandas as pd
from tqdm import tqdm 

import utils

def mlp_train(
    network,
    loss_func,
    data,
    config
):
    # Report of training
    report = pd.DataFrame({
        "epoch": [],
        "batch_id": [], 
        "loss": [],
        "phase": []
    })
    
    # Loss Tracking
    loss_tracker = utils.AverageMeter()
    
    # Accuracy Meter
    accuracy_meter = utils.AverageMeter()
    
    # training for some epochs
    for epoch in range(config.num_epochs):
        
        # walking on batches!!!
        with tqdm( range(data.get_num_batches()) ) as t_batches:
            
            t_batches.set_description(f"Training @ Epoch: {epoch}")
            
            for ix in t_batches:    
                # Getting data of current batch
                X , Y_target = data.get_batch(ix)
                
                # obtaining output
                Y = network.forward(X)
                
                # if np.max(Y) == np.inf:
                    # import pdb; pdb.set_trace()
                
                # calculating loss value
                loss = loss_func.forward(Y,Y_target)
                
                # updating loss tracker
                loss_tracker.update([loss.item()])
                
                # calculating accuracy
                batch_accuracy = utils.accuracy(Y, Y_target)
                
                # Updating ongoing accuracy
                accuracy_meter.update([batch_accuracy.item()])
                
                # calculating gradient of loss with respect to output (dloss/dy)
                dloss_dY = loss_func.backward(Y,Y_target)
                
                # Applying backward path and updating network
                network.backward(dloss_dY,X)
                
                t_batches.set_postfix(
                    accuracy = accuracy_meter.avg,
                    loss = loss_tracker.avg
                )
                
                # Adding current batch report to the training report
                batch_report = pd.DataFrame({
                                "epoch": [epoch],
                                "batch_id": [ix], 
                                "loss": [loss],
                                "phase": ["train"]
                            })
                report = pd.concat([report, batch_report])
                
    return network, report
