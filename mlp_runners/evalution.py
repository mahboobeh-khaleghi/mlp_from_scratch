import numpy as np
import pandas as pd
import dataloaders
import losses
import nets

def mlp_eval(
    network,
    loss_func,
    data,
    config
):
    # Report of evaluation
    report = pd.DataFrame({
        "batch_id": [], 
        "loss": [],
        "phase": []
    })
    
    # walking on batches!!!
    for ix in range(data.get_num_batches()):
        
        # Getting data of current batch
        X , Y_target = data.get_batch(ix)
        
        # obtaining output
        Y = network.forward(X)
        
        # calculating loss value
        loss = loss_func.forward(Y,Y_target)
        
        # Adding current batch report to the training report
        batch_report = pd.DataFrame({
                        "batch_id": [ix], 
                        "loss": [loss],
                        "phase": ["train"]
                    })
        report = report.append(batch_report , ignore_index = True)
        
    return network, report
