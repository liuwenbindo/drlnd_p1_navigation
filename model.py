import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers_config_dict, layer_config_sequence, kernel_size=(3,3)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_config_dict = layers_config_dict
        self.layer_config_sequence = layer_config_sequence
        self.kernel_size = kernel_size
        self.layer_list = nn.ModuleList()
        last_layer_node_ct = state_size
        for l in range(len(self.layer_config_sequence)):
            if "CONV" in self.layer_config_sequence[l]:
                pass
            elif "RELU" in self.layer_config_sequence[l]:
                pass
            elif "FC" in self.layer_config_sequence[l]:
                if l == len(self.layer_config_sequence)-1:
                    self.layer_list.append(nn.Linear(self.layer_config_dict[self.layer_config_sequence[l-1]], action_size))
                elif l == 0:
                    self.layer_list.append(nn.Linear(state_size, self.layer_config_dict[self.layer_config_sequence[l]]))
                else:
                    self.layer_list.append(nn.Linear(self.layer_config_dict[self.layer_config_sequence[l-1]], self.layer_config_dict[self.layer_config_sequence[l]]))
                
    def forward(self, state):
        """Build a network that maps state -> action values."""
        ss = state
        for l in range(len(self.layer_list)):
            ss = self.layer_list[l](ss)
            if l < len(self.layer_list)-1:
                ss = F.relu(ss)
        return ss
    
#     def train(self):
#         pass
       
#     # Get action values from state
#     def get_item(self, state):
#         pass
    
#     def grad_descent(self, optimizer):
#         # Calculate loss, and use optimizer to minimize loss
#         pass