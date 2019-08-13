import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import fgn_helper_lib as fgnh

def FGSM_attack_untargeted(model, input_data, max_noise, loss_func, step_size, data_bounds,
                           steps=1, confidence_req=0.5,
                           **kwargs):
    # fast gradient sign method
    # returns an adversarial noise for this input (and the adversarial input that goes with the noise)
    # assumes the argmax index of the output is the prediction
    # currently only uses inf-norm as measure of the noise amplitude
    # in original paper, steps = 1!
    # changes to original paper/code: 
    # - attempt to increase confidence on wrong class by changing step size and direction
    # - if zero grad found, takes random step
    
    # model: a pytorch model that outputs a raw vector for a N-class prediction
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model is not pytorch module")
    # input_data: a tensor accepted by the model 
    if not isinstance(input_data, torch.Tensor):
        raise TypeError("input is not a pytorch tensor")
    
    # max_noise: positive float, maximum change allowed in each input dimension
    
    # loss_func: the loss function used to compute the adversarial gradients
    
    # step_size: espilon on rest of literature 
    
    # data_bounds: tuple or list. boundaries for data values (to be updated to a min/max for each dim? )
    data_min, data_max = min(data_bounds), max(data_bounds)
    
    # steps: integer number of steps of gradient compute
    
    # confidence_req: require that the confidence of the adversarial pred be higher than this
    
    ### kwargs    
    # verbose: boolean, used to print training stats
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    # device: a pytorch device, used to tell where to run the model
    if 'device' in kwargs:
        # check device type
        if not isinstance(kwargs['device'], torch.device):
            raise TypeError("device is not pytorch device")
        # give warning
        if verbose: print("Warning: device specified. This might change model and input location (cuda<->cpu)")
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
#     input_data = input_data.to(device)
    
    # set model to eval mode
    model.eval()
    
    # get prediction for input (int)
    # this might even be the wrong class, but the attack aims to change this prediction
    orig_class = fgnh.get_class_from_pred(model, input_data, **kwargs)
   
    ### start of attack code
    cur_best_confidence = -1.0
    print_once = True
        
    # variables to return
    adv_input = input_data
    cur_best_adv = adv_input
    cur_best_noise = torch.zeros_like(cur_best_adv)
    # current adv_input (this is the leaf var for which we keep the grads)
    cur_adv_input = Variable(input_data, requires_grad=True)
    
    # start of attack
    # (wastes the final step of the loop, since it doesnt check after update)
    for step in range(steps):
#         if verbose: print("Step:", step)
        # reset the gradients
        zero_gradients(cur_adv_input)

        # compute model output for current adv_input
        cur_out = model(cur_adv_input)
        # set current prediction
        cur_pred = cur_out.data.max(1)[1]
        cur_class = cur_pred.cpu().numpy()[0]
#         if verbose: print("Current prediction:", cur_pred)

        confidence = torch.softmax(cur_out,1).max(1)[0].detach().cpu().numpy()[0]
        
        # check if already successful
        if (cur_class != orig_class) and (confidence >= confidence_req) :
            if verbose: print("Early stopping at step {} with confidence {}:".format(step, confidence))
            cur_best_confidence = confidence
            cur_best_adv = adv_input
            cur_best_noise = adv_noise
            # exit 'for step' loop
            break
            
        # update whenever we find new best confidence in the 'wrong' class
        if (cur_class != orig_class):
            if (confidence > cur_best_confidence):
                cur_best_confidence = confidence
                cur_best_adv = adv_input
                cur_best_noise = adv_noise
                if verbose and not print_once:
                    print("New best found at step {} with confidence {}:".format(step,cur_best_confidence))
            else:
                # update step size to avoid stepping too far
                step_size = 0.99*step_size #crude
            
        # if not successful yet, update steps, noise, and adv input

        # compute current loss, to be maximized for untargeted attack
        cur_loss = loss_func(cur_out, cur_pred)
        # apply backprob
        cur_loss.backward()
        # sign of the grad * step size
        normed_grad = step_size * torch.sign(cur_adv_input.grad.data)
        
        # if grad is zero, add tiny noise
        if (torch.abs(normed_grad).max()<=1e-32):
#             if verbose: print("Grad zero found. Taking Random Step")
            normed_grad =  step_size * torch.sign(2.0*torch.rand_like(normed_grad)-1.0)

        #if we have already have the wrong class, but not enough confidence, move with the gradient
        if (cur_class != orig_class):
            if verbose and print_once: 
                print("Found wrong class {} at step {} with confidence {}.".format(cur_class ,step, confidence))
                print("Attempting to increase confidence")
                print_once = False
            
            # step WITH the gradient
            step_adv = cur_adv_input.data - normed_grad
            
        else:
            # still original class , take one step AWAY (opposite the gradient, away from current class)
            step_adv = cur_adv_input.data + normed_grad

        # compute adv noise
        adv_noise = step_adv - input_data
        # clip to max noise
        adv_noise = torch.clamp(adv_noise, min=-max_noise, max=max_noise)

        # compute current adv input
        adv_input = input_data + adv_noise
        # clip to max and min value of dataset
        adv_input = torch.clamp(adv_input, data_min, data_max)
        # cur_adv input for grad compute 
        cur_adv_input.data = adv_input
        
#         # update whenever we find new best confidence in the 'wrong' class
#         if (confidence > cur_best_confidence) and (cur_class != orig_class):
#             cur_best_confidence = confidence
#             cur_best_adv = adv_input
#             cur_best_noise = adv_noise
#             if verbose: print("New best confidence found at step {}:".format(step), cur_best_confidence)
#         else:
#             # update step size
#             step_size = 0.99*step_size #crude

    # return dict of results for analysis
    results = {'steps':step, 'confidence':cur_best_confidence}
        
    return cur_best_adv.cpu(), cur_best_noise.cpu(), results    