import sys
sys.path.append('../')
from util.header import *


class AttrLayer(nn.Module):
    def __init__(self, opt, inplanes=1024):
        super(AttrLayer,self).__init__()
        self.opt = opt
        attr_layer = []
        for idx, num in enumerate(opt.num_attrs):
            attr_layer += [nn.Linear(inplanes, num)]
        self.attr_layer =nn.ModuleList(attr_layer)
        
    def forward(self,features):
        output = []
        for layer in self.attr_layer:
            output.append(layer(features))
        return output
        
