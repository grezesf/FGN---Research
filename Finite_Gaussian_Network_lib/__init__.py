## Matlab style file organization, one function per file

# import sys
# sys.path.append('/home/felix/Research/Adversarial Research/FGN---Research/Finite_Gaussian_Network_lib')

from .FGN_layer import FGN_layer
from .Feedforward_FGN_net import Feedforward_FGN_net
from .Feedforward_Classic_net import Feedforward_Classic_net
from .convert_classic_to_fgn import convert_classic_to_fgn
from .convert_fgn_to_classic import convert_fgn_to_classic
from .sigmas_loss import sigmas_loss
from .fgn_cross_ent_loss import fgn_cross_ent_loss
from .def_fgn_cross_ent_loss import def_fgn_cross_ent_loss
from .adjust_sigma_pred_func import adjust_sigma_pred_func
from .FGSM_attack_untargeted import FGSM_attack_untargeted
from .FGSM_attack_targeted import FGSM_attack_targeted
from .full_FGSM_attack import full_FGSM_attack
from .plot_centers_histories import plot_centers_histories
from .plot_sigmas_histories import plot_sigmas_histories
from .get_fgsm_attack_vectors import get_fgsm_attack_vectors
from .FGN_Conv1d import FGN_Conv1d
from .convert_layer_conv1D_to_fgn import convert_layer_conv1D_to_fgn
from .perform_attack import perform_attack
__version__ = '0.01'


### deprecated functions
# from .Feedforward_Hybrid_First_net import Feedforward_Hybrid_First_net
# from .Feedforward_Hybrid_Last_net import Feedforward_Hybrid_Last_net
# from .Feedforward_First_Centers_net import Feedforward_First_Centers_net