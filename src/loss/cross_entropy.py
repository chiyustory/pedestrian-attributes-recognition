import sys
sys.path.append('../')
from util.header import *

class CrossEntropy(nn.Module):
    def __init__(self, opt):
        super(CrossEntropy,self).__init__()
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, outputs, targets):
        loss = 0
        loss_ls = list()    
        for attr in range(len(self.opt.num_attrs)):
            # pdb.set_trace()
            sub_loss = self.criterion(outputs[attr], targets[:, attr])
            loss_ls.append(sub_loss)
            loss += sub_loss
        return loss,loss_ls