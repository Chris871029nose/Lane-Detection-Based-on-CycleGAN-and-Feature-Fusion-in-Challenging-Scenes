import torch
from models import networks

model = networks.ResnetGenerator(3,3)

def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
            
state_dict = torch.load('latest_net_G_A.pth', map_location=torch.device('cuda:0'))
#state_dict = torch.load('latest_net_G_A.pth')
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata
for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    __patch_instance_norm_state_dict(state_dict, model, key.split('.'))
model.load_state_dict(state_dict)
