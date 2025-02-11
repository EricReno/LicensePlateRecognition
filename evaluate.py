import os
import cv2
import torch
import numpy as np
from config import parse_args
from model.build import build_net
from dataset.build import build_dataset
from utils.utils import strLabelConverter

class Evaluator():
    """ VOC AP Evaluation class"""
    def __init__(self,
                 device,
                 dataset,
                 converter,
                 image_h,
                 image_w,
                 visualization) -> None:
        
        self.device = device
        self.dataset = dataset
        self.converter = converter
        self.input_width = image_w
        self.input_height = image_h
        self.visualization = visualization
        self.mean = np.array(0.588, dtype=np.float32)
        self.std = np.array(0.193, dtype=np.float32)
        
    def eval(self, model):
        sum = 0
        n_correct = 0
        for i in range(len(self.dataset)):
            label = self.dataset.pull_anno(i)
            image = self.dataset.pull_image(i)
        
            # preprocess
            img_h, img_w, _ = image.shape
            img = cv2.resize(image, (self.input_width, self.input_height)).astype(np.float32)
            img = (img/255. - self.mean) / self.std
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img).contiguous().float()
            img = img.unsqueeze(0).to(self.device)

            # infer
            outputs = model(img)
            
            # postprocess
            _, outputs = outputs.max(2)
            outputs = outputs.transpose(1, 0).contiguous().view(-1)
            output_str = self.converter.decode(outputs.data, torch.IntTensor([outputs.size(0)]).data, raw=False)
                        
            if self.visualization:
                # TODO  Visualization Debug
                text_image = np.zeros((40, self.input_width, 3), dtype=np.uint8)
                cv2.putText(text_image, output_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                
                show_image = np.concatenate((image, text_image), axis=1)
                cv2.imshow('1', show_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            for pred, target in zip(output_str, label):
                sum += 1
                if pred == target:
                    n_correct += 1
            print('Inference: {} / {}'.format(i+1, len(self.dataset)), end='\r')
            
        accuracy = n_correct / sum
        
        return accuracy

def build_eval(args, dataset, device):
    converter = strLabelConverter(args.class_names[1:])
    evaluator = Evaluator(
        device   = device,
        dataset  = dataset,
        converter = converter,
        image_h = args.image_size_h,
        image_w = args.image_size_w,
        visualization = args.eval_visualization)
    
    return evaluator
    
if __name__ == "__main__":
    args = parse_args()
    args.resume_weight_path = "None"
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('use cuda')
    else:
        device = torch.device('cpu')

    val_dataset = build_dataset(args, is_train=False)

    model = build_net(args, device, export = False)
    model = model.eval()

    state_dict = torch.load(f = os.path.join('log', args.model_weight_path), 
                            map_location = 'cpu', weights_only = False)
    model.load_state_dict(state_dict["model"])
    print('acc:', state_dict['acc'])

    # VOC evaluation
    evaluator = build_eval(args, val_dataset, device)
    acc = evaluator.eval(model)
    print(acc)