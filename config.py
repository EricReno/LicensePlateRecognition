import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='You Only Look Once')
    parser.add_argument('--cuda', default=True,   type=bool)
    parser.add_argument('--num_workers',  default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size used during training (per GPU).')
    parser.add_argument('--epochs_total', default=300, type=int, help='Total number of training epochs.')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='Number of warm-up epochs.')
    parser.add_argument('--save_checkpoint_epoch', default=0, type=int, help='Epoch interval to save model checkpoints.')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--lr', default=0.0001, type=float, help='Base learning rate.')
    parser.add_argument('--momentum', default=0.0, type=float, help='Momentum factor for SGD optimizer.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay factor for regularization.')
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--lr_scheduler', default='linear', type=str)  

    # # Data settings
    parser.add_argument('--data_root', default=['data/Public'], help="format: ['data/Public', 'data/Private']")
    parser.add_argument('--image_size_h', default=48, type=int, help='Input image size Height')
    parser.add_argument('--image_size_w', default=168, type=int, help='Input image size Width')
    parser.add_argument('--num_classes', default=78, type=int, help='Number of object classes.')
    parser.add_argument('--class_names', default="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品")

    # Model settings
    parser.add_argument('--model', default='mynet_m', choices=['mynet_s', 'mynet_m', 'mynet_b'])

    # Eval settings
    parser.add_argument('--model_weight_path', default='15.pt', type=str, help='Path to the initial model weights.')
    parser.add_argument('--resume_weight_path', default='None', type=str, help='Path to the checkpoint from which to resume training.')
    parser.add_argument('--eval_visualization', default=False, type=bool, help='Whether to visualize the evaluation results.')

    return parser.parse_args()