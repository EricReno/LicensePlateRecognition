import onnxruntime
import numpy as np
from PIL import Image
from inference_box import infer as infer_box
colors = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
def setup_inference(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})] if args.cuda else [('CPUExecutionProvider', {})]
    return onnxruntime.InferenceSession(args.rec_onnx, providers=providers)
def decodePlate(index):
    pre = 0
    n_index = []
    for i in range(len(index)):
        if index[i] != 0 and index[i] != pre:
            n_index.append(i)
        pre = index[i]
    return n_index

def preinfer(image, image_size):
    mean_value, std_value = (0.588, 0.193)
    output = np.array((Image.fromarray(image)).resize((image_size[0], image_size[1])))
    
    output = output.transpose([2, 0, 1]).astype(np.float32)
    output = (output / 255. - mean_value) / std_value
    output = np.expand_dims(output, 0)
    return  image, output

def postinfer(preds, color_preds):
    def stable_softmax(x):
        """数值稳定的 softmax 实现"""
        x = np.array(x)
        # 防止溢出，先减去每行的最大值
        x_max = np.max(x, axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)  # 减去最大值以稳定计算
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        # 避免除以零
        x_sum = np.where(x_sum == 0, 1e-9, x_sum)
        return x_exp / x_sum
    
    color_preds = stable_softmax(color_preds)
    color_conf = np.max(color_preds, axis=-1)  # 最大概率
    color_index = np.argmax(color_preds, axis=-1)
    color_conf, color_index = float(color_conf), int(color_index)
    preds = stable_softmax(preds)
    pred_conf = np.max(preds, axis=-1)[0]  # 最大概率
    pred_index = np.argmax(preds, axis=-1)[0]  # 最大概率对应的索引
    new_index = decodePlate(pred_index)
    pred_conf = pred_conf[new_index]
    pred_index = pred_index[new_index]
    plate = "".join([plateName[i] for i in pred_index])
    return plate, pred_conf, colors[color_index], color_conf

def run(args, image):
    labels, _, bboxes = infer_box(args, image)
    session = setup_inference(args)
    
    for index, box in enumerate(bboxes):
        box = [int(x) for x in box]
        roi_img = image[box[1]+2:box[3], box[0]:box[2]]
        if int(labels[index]):
            h,w,c = roi_img.shape
            img_upper = roi_img[0:int(5/12*h),:]
            img_lower = roi_img[int(1/3*h):,:]
            img_upper = np.array((Image.fromarray(img_upper)).resize((img_lower.shape[1], img_lower.shape[0])))
            roi_img = np.hstack((img_upper,img_lower))
            
        roi_img, infer_input = preinfer(roi_img, args.roi_size)
        preds, color_preds = session.run(['feature', 'color'], {'input': infer_input})
        plate, pred_c, color, color_c = postinfer(preds, color_preds)
    
        result = plate + ' ' + color
        
        return (result)
    
if __name__ == "__main__":
    import argparse
    from PIL import Image
    from io import BytesIO
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, help='source')
    parser.add_argument('--rec_onnx', default='platerec.onnx', help='model.pt path(s)')
    parser.add_argument('--det_onnx', default='platedet.onnx', help='model.pt path(s)')
    parser.add_argument('--image_size', default=640, type=int, help='Input image size.')
    parser.add_argument('--confidence', default=0.5, type=int, help='Input image size.')
    parser.add_argument('--nms_thresh', default=0.1, type=int, help='Input image size.')
    parser.add_argument('--roi_size', default=[168, 48], type=int, help='Input image size.')
    parser.add_argument('--class_names', default=['plate', 'plate_d'], type=int, help='Input image size.')
    
    args = parser.parse_args()
    
    with open('imgs/粤NBM959.png', 'rb') as image_file:
        file_bytes = image_file.read()
    image = np.array(Image.open(BytesIO(file_bytes)).convert("RGB"))[:,:, ::-1]
    
    print(run(args, image))
    