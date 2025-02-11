import torch
import torch.nn as nn
from config import parse_args
import torch.nn.functional as F

import os
import sys
cur = os.path.dirname(os.path.abspath(__file__))
pro_path = os.path.abspath(os.path.join(cur, '..'))
sys.path.append(pro_path)

class MyNet(nn.Module):
    def __init__(self, 
                 device,
                 model_cfg,
                 num_classes,
                 export = False
                 ):
        super(MyNet, self).__init__()
        
        self.export = export
        
        self.features = self.make_layers(model_cfg, batch_norm=True)
        
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)

        self.cls = nn.Conv2d(model_cfg[-1], num_classes, 1, 1)
        
    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for i, l in enumerate(cfg):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, out_channels=cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                
                in_channels = cfg[i]
            else:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, out_channels=cfg[i], kernel_size=3, padding=1, stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    
                    in_channels = cfg[i]
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.loc(x)
        x = self.cls(x)
        if self.export:
            x = x.squeeze(2)
            x = x.transpose(2,1)
            output = x.argmax(dim=2)
            return output
        else:
            b, c, h, w = x.size()
            x = x.squeeze(2)
            x = x.permute(2, 0, 1)
            output = F.log_softmax(x, dim=2)
            return output

if __name__ == "__main__":
    import time
    from thop import profile
    args = parse_args()

    input = torch.randn(1, 3, 48, 168)
    model = MyNet(
        device = 'cpu',
        model_cfg = [16,16,32,32,'M',64,64,'M',96,96,'M',128,256],
        num_classes = 78,
        )

    t0 = time.time()
    outputs = model(input)
    t1 = time.time()
    print('Time:', t1-t0)
    
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))