import os
import torch
import torch.optim as optim

def build_optimizer(args, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(args.optimizer))
    print('--base lr: {}'.format(args.lr))
    print('--momentum: {}'.format(args.momentum))
    print('--weight_decay: {}'.format(args.weight_decay))
    
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )
    
    start_epoch = 0
    if args.resume_weight_path and args.resume_weight_path != 'None':
        ckpt_path = os.path.join('log', args.resume_weight_path)
        checkpoint = torch.load(ckpt_path, weights_only=False)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint.pop("optimizer")
            print('Load optimizer from the checkpoint: ', args.resume_weight_path)
            optimizer.load_state_dict(checkpoint_state_dict)
            start_epoch = checkpoint.pop("epoch") + 1
            del checkpoint, checkpoint_state_dict
        except:
            print("No optimzier in the given checkpoint.")
    
    return optimizer, start_epoch