import torch
import torch.nn as nn
import numpy as np

class FGN_Conv1d(nn.Module):
    # the fgn version of the convolutional 1d layer
    # tries to handle as many params as Conv1D in the same manner
    
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, padding=0, padding_mode='zeros',
                 covar_type='diag', ordinal=2.0,
                 **kwargs):
        super(FGN_Conv1d, self).__init__()
        # the associated conv1d layer
        self.Conv1d = nn.Conv1d(in_channels=in_channels ,out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, dilation=dilation, padding=padding, padding_mode=padding_mode,
                                **kwargs)
        # input channels
        self.in_channels = in_channels
        # output dimensions 
        self.out_channels = out_channels
        # size of the kernel (analog to the input dimension)
        self.kernel_size = kernel_size
        # stride step
        self.stride = stride
        # dilation between kernel inputs
        self.dilation = dilation
        # padding params
        self.padding = padding
        self.padding_mode = padding_mode
    
        # covariance type for the gaussian: one of 'sphere', 'diag', 'full'
        self.covar_type = covar_type
        # ordinal used for the norm (distance to centers) computation 
        # (1=diamond, 2=euclidean, 3->float('inf')=approach manhattan)
        self.ordinal = ordinal
        # centers of FGNs  
        self.centers = nn.Parameter(torch.Tensor(out_channels, kernel_size), requires_grad=True)
        # size/range of FGNs
        # inverse covariance will actually be used
        if covar_type == 'sphere':
            self.inv_covars = nn.Parameter(torch.Tensor(out_channels,), requires_grad=True)
        elif covar_type == 'diag':
            self.inv_covars = nn.Parameter(torch.Tensor(out_channels, kernel_size,), requires_grad=True)
        elif covar_type == 'full':
            raise Exception('full gaussian type not supported (and unlikely to be considering computation cost)')
        elif covar_type == 'chol':
            raise Exception(' Cholesky full gaussian type not supported (and unlikely to be considering computation cost)')
        else:
            # error
            raise TypeError('covar_type not one of [\'sphere\', \'diag\', \'full\', \'chol\']')
        
        # minimum sigma/range of neurons
        self.eps = 1e-8 
        
        # parameter init call
        self.reset_parameters()
    
    # parameter init definition
    def reset_parameters(self):
        # centers init, assuming data normalized to mean 0 (not necessarily true after first layer)
        nn.init.normal_(self.centers, mean=0.0, std=0.1)
        # sigmas init, to be researched further
        s = self.kernel_size

#         self.sigmas.data.uniform_(s-0.5, s+0.5)
        self.inv_covars.data.uniform_(3.0/(s+0.5), 3.0/(s-0.5))

        
    def forward(self, inputs, prev_g = None):
        # compute conv output
        c = self.Conv1d.forward(inputs)
        
        # transform the inputs per in/out channels/stride/kernelsize
        batch_size, _, input_len = inputs.size()
        # conv1d padding only does zero padding, but nn.pad does constant
        if self.padding_mode == 'zeros':
            padding_mode_passed = 'constant'
        else:
            padding_mode_passed = self.padding_mode

        # (conv1d doesnt handle different sized paddings on each sides)
        # even though nn.functional.pad does, the code below should future proof it in case they move to left/right pads
        padding_passed = (self.padding, self.padding)
        
        # total numpber of outputs
        num_outputs = int(np.floor((input_len+sum(padding_passed)-self.dilation*(self.kernel_size-1)-1)/self.stride+1))
        # stride the inputs
        strided_inputs = torch.as_strided(input=nn.functional.pad(inputs, pad=padding_passed, mode=padding_mode_passed), 
                                  size=(batch_size,
                                        self.in_channels,
                                        num_outputs,
                                        self.kernel_size, 
                                       ),
                                  stride=((input_len+sum(padding_passed))*(self.in_channels),
                                          input_len+sum(padding_passed), 
                                          self.stride, 
                                          self.dilation))
        
        # distance to centers
        dists = strided_inputs.view(batch_size, self.in_channels, 1, num_outputs, self.kernel_size) - self.centers.view(1, 1, self.out_channels, 1, self.kernel_size)
            
        # computation of gaussian component
        if self.covar_type=='sphere':
            # distance to centers (sum along 1)
            # with ordinal applied,
            # and small value added to prevent problems when ord<1
            # summed along channels (dim 1)
            g = torch.sum( torch.pow( torch.abs(dists)
                                           + 1e-32*float(self.ordinal<1.0),
                                           self.ordinal),
                                dim=(1,4))


            # remember: self.eps defines minimum neuron range (prevents zero range neuron, which can't later grow if needed)
            # with inv_covar applied,
            g = g*(torch.clamp(self.inv_covars.view(1, self.out_channels, 1), max=1.0/self.eps)**2)

        elif self.covar_type=='diag':
            # distance to centers,
            # with inv_covar applied,
            # with ordinal applied,
            # and small value added to prevent problems when ord<1
            g = torch.sum( torch.pow( torch.abs(dists) * 
                                          torch.pow(
                                              torch.clamp(self.inv_covars.view(1, 1, self.out_channels, 1, self.kernel_size), max=1.0/self.eps)**2, 
                                              1./self.ordinal)
                                          + 1e-32*float(self.ordinal<1.0), 
                                          self.ordinal),
                                dim=(1,4))

        else:
            # this should never happen
             raise Exception('Something went wrong with covar_type')
                
        # apply exponential
        g = torch.exp(-g)
            
        # apply prev_g gate if needed
        # in the future: preg_g can have shape: single scalar for batch, per inputs, per channel, per value 
        # for now, same shape input
        if prev_g is not None:
            assert prev_g.size() == inputs.size()
            
            # pad_mode is always zeros
            prev_g_strided = torch.as_strided(input=nn.functional.pad(prev_g, pad=padding_passed, mode='constant'), 
                                              size=(batch_size,
                                                    self.in_channels,
                                                    num_outputs,
                                                    self.kernel_size, 
                                                   ),
                                              stride=((input_len+sum(padding_passed))*(self.in_channels),
                                                      input_len+sum(padding_passed), 
                                                      self.stride, 
                                                      self.dilation))

            # max along channels, kernels
            max_prev_g_strided,_ = torch.max(prev_g_strided, dim=3)
            max_prev_g_strided,_ = torch.max(max_prev_g_strided, dim=1)
            
            # compare with current g
            g = torch.maximum(g, max_prev_g_strided.unsqueeze(1))
            
        # combine conv with gaussian
        res = c*g

        # pass both activity and gaussian component
        return res, g