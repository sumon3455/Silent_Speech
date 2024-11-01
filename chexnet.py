import numpy as np
import torchvision 
import torch
import torch.nn as nn


def chexnet(parentdir):
 
    nnClassCount = 14
    nnIsTrained = True
    model = DenseNet121(nnClassCount, nnIsTrained).cuda()
    model = torch.nn.DataParallel(model).cuda()
    # checkpoint = parentdir + 'models/m-25012018-123527.pth.tar'
    checkpoint = parentdir + 'models/chexnet.pth.tar'
    checkpoint = torch.load(checkpoint)

    import re
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    state_dict = checkpoint['state_dict']
    remove_data_parallel = False # Change if you don't want to use nn.DataParallel(model)
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel:   
            del state_dict[key]

    model.load_state_dict(state_dict)

    return model 


class DenseNet121(nn.Module):
    
    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x



