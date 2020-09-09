from datetime import datetime


def train_and_save_model(full_save_path, model_type, train_dataloader, val_dataloader, **kwargs):
    
    # given a model type ('classic' or 'fgn'), and dataloaders, trains the model 
    # and saves model at save_path, and results (params, histories) in same directory
    # kwargs are passed to model definition and training
    
    # define model
    if model_type == 'classic':
        model = Feedforward_Classic_net(**kwargs)
    elif model_type == 'fgn':
        model = Feedforward_FGN_net(**kwargs)
    else:
        # error
        raise TypeError("model_type is classic or fgn")
    
    # directory for saving
    save_dir = os.path.dirname(os.path.abspath(save_path))
    
    # save initial model (pre-trained)
    
    
    # train the model