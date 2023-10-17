import nets
import losses
import dataloaders
import mlp_runners
import utils

def get_loss(loss_type):
    if loss_type.lower() == "mse":
        loss_func = losses.L2Loss()
    elif loss_type.lower() == "softmax_cross_entropy":
        loss_func = losses.SoftmaxCrossEntropy()
    else:
        print("EROR! undefined loss function")
        exit()
    return loss_func

def main(args):
    # read config 
    config = utils.read_config(args.config_path)
    
    # Instanciating a dataloader
    data = dataloaders.Cifar10DataLoader(
        pickles_path = config.pickles_path,
        batch_size = config.batch_size,
        preprocessing_method = config.preprocessing_method
    )
    
    # Constructing a MLP network
    network = nets.MLP(
        n_nodes = config.n_nodes,
        act_func_list = config.act_func_list,
        lr = config.lr,
        l2_reg_coef = config.l2_reg_coef
    )
    
    # Defining the loss function
    loss_func = get_loss(config.loss_type)
    
    # Training the network
    network, train_report = mlp_runners.train(
        network = network,
        loss_func = loss_func,
        data = data,
        config = config
    )
    
    # Saving training report
    train_report.to_csv(config.report_path)
    

if __name__ == "__main__":
    args = utils.get_args()
    main(args)