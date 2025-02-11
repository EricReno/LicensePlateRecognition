import os
import time
import torch
import numpy
from torch.utils.tensorboard import SummaryWriter

from config import parse_args
from evaluate import build_eval
from model.build import build_net
from utils.loss import build_loss
from utils.flops import compute_flops
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lambda_lr_scheduler
from dataset.build import build_dataset, build_dataloader
from utils.utils import strLabelConverter

def train():
    args = parse_args()
    writer = SummaryWriter('log')
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    converter = strLabelConverter(args.class_names[1:])

    # ---------------------------- Build --------------------------
    val_dataset = build_dataset(args, is_train=False)
    train_dataset = build_dataset(args, is_train=True)
    train_dataloader = build_dataloader(args, train_dataset)

    model = build_net(args, device, export = False)
    compute_flops(model, args.image_size_h, args.image_size_w, device)
          
    loss_function =  build_loss(args, device)
    
    evaluator = build_eval(args, val_dataset, device)
    
    optimizer, start_epoch = build_optimizer(args, model)
    lr_scheduler, lf = build_lambda_lr_scheduler(args, optimizer)
    if args.resume_weight_path and args.resume_weight_path != 'None':
        lr_scheduler.last_epoch = start_epoch - 1
        optimizer.step()
        lr_scheduler.step()
    
    # ----------------------- Train --------------------------------
    print('==============================')
    max_acc = 0
    start = time.time()
    for epoch in range(0, args.epochs_total):
        model.train()
        train_loss = 0.0
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * len(train_dataloader)
           
            images = images.to(device)
            
            ## forward
            outputs = model(images)
            
            text, length = converter.encode(targets)
            pred_size = torch.IntTensor([outputs.size(0)] * args.batch_size)
            loss = loss_function(outputs, text, pred_size, length)
            
            loss.backward()
            
            # optimizer.step
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(train_dataloader), optimizer.param_groups[0]['lr'], loss))

        lr_scheduler.step()

        # chech model
        model_eval = model
        model_eval.eval()
        # save_model
        if epoch >= args.save_checkpoint_epoch:
            ckpt_path = os.path.join(os.getcwd(), 'log', '{}.pt'.format(epoch))
            if not os.path.exists(os.path.dirname(ckpt_path)):
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                acc = evaluator.eval(model_eval)
            writer.add_scalar('acc', acc, epoch)

            if acc > max_acc:
                torch.save({
                        'model': model_eval.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'acc':acc,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_acc = acc
        
if __name__ == "__main__":
    train()