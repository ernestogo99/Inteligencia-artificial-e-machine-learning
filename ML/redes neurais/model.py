import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer=nn.Linear(in_features=1,out_features=1,bias=True)

    
    def forward(self,x):
        out=self.input_layer(x)
        return out