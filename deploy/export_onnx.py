import os
import sys
import onnx
import torch
sys.path.append('../')
from config import parse_args
from model.build import build_net

def export(input, model, weight_name):
    weight_path = os.path.join(os.getcwd(), weight_name)
    pt_onnx = weight_path.replace('.pt', '.onnx')

    state_dict = torch.load(weight_path, 
                            map_location = 'cpu', 
                            weights_only = False)
    model.load_state_dict(state_dict.get("model", state_dict))
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            pt_onnx,
            opset_version=11,
            input_names=['input'],
            output_names=['output'])

    # 添加中间层特征尺寸
    onnx_model = onnx.load(pt_onnx) 
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), pt_onnx)

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect")
    else: 
        print("Model correct")

if __name__ == "__main__":
    args = parse_args()
    args.resume_weight_path = 'None'
    
    x = torch.randn(1, 3, 48, 168)
    
    model = build_net(args, torch.device('cpu'), export = True)
    model = model.eval()
    
    export(x, model, args.model_weight_path)
     

    # colors = ['黑色', '蓝色', '绿色', '白色', '黄色']
    # plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

    

    # # ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    # # cfg, model_state = ckpt['cfg'], ckpt['state_dict']
    
        
    
    # # model = OCRNet(cfg=cfg,
    # #                num_classes=len(plateName),
    # #                export=True,
    # #                color_num=len(colors))
    # # model.load_state_dict(model_state, strict=True)
    
    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         x,
    #         "platerec.onnx",
    #         opset_version=11,
    #         input_names=['input'],
    #         output_names=['feature', 'color'])
        