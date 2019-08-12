import torch

def get_class_from_pred(model, input_data, classes=None, **kwargs):
    # given a model and an input, returns the class of the prediction
    # assuming the argmax index of the output is the prediction
    
    # model: a pytorch model that outputs a raw vector for a N-class prediction
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not pytorch module")
    # input_data: a tensor accepted by the model 
    if not isinstance(input_data, torch.Tensor):
        raise TypeError("input is not a pytorch tensor")
    # classes: either an int N for number of classes, and array with class names
    if classes!=None and not isinstance(classes, list):
         raise TypeError("classes must be a list")
    # convert to list if needed
    if classes == None:
        # make sure your model has 'out_feats' attribut
        classes = range(model.out_feats)
    
    ### kwargs    
    # device: a pytorch device, used to tell where to run the model
    if 'device' in kwargs:
        # check device type
        if not isinstance(kwargs['device'], torch.device):
            raise TypeError("device is not pytorch device")
        # give warning
        print("Warning: device specified. This might change model and input location (cuda<->cpu)")
        device = kwargs['device']
        change_device = True
    else:
        # get device from module (will run into probs if on multiple gpus)
        device = next(model.parameters()).device
        change_device = False
    
    # send model to device
    if change_device:
            model.to(device)
            
    # send input to device
    input_data = input_data.to(device)
    
    # make prediction 
    pred = model(input_data)
    
    # get index of max of prediction
    pred_class = pred.data.max(1)[1].cpu().numpy()[0]
    
    # return the name of the predict class
    return classes[pred_class]