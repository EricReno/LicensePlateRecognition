import os
import torch
import torch.nn as nn
from .net import MyNet

def build_net(args, device, export):
    print('==============================')
    print('Build Model: {}'.format(args.model))
    print('')
    
    model_cfg = {
    'mynet_s' : [8,8,16,16,'M',32,32,'M',48,48,'M',64,128], #small model
    'mynet_m' : [16,16,32,32,'M',64,64,'M',96,96,'M',128,256], #medium model
    'mynet_b' : [32,32,64,64,'M',128,128,'M',196,196,'M',256,256] #big model
    }
        
    model = MyNet(
        device = device,
        num_classes = args.num_classes,
        model_cfg = model_cfg[args.model],
        export = export).to(device)
    
    # -------------- Initialize YOLO --------------
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))

    # keep training
    if args.resume_weight_path and args.resume_weight_path != "None":
        ckpt_path = os.path.join('log', args.resume_weight_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint['model']
            print('Load model from the checkpoint: ', ckpt_path)
            model.load_state_dict(checkpoint_state_dict, strict=False)
            
            del checkpoint, checkpoint_state_dict
        except:
            print("No model in the given checkpoint.")

    return model