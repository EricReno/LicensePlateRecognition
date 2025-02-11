import onnxruntime
import numpy as np
from PIL import Image

def setup_inference(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})] if args.cuda else [('CPUExecutionProvider', {})]
    return onnxruntime.InferenceSession(args.det_onnx, providers=providers)

def xywh2xyxy(bboxes):
    pred_wh = bboxes[..., 2:]
    pred_ctr = bboxes[..., :2]
    pred_x1y1 = pred_ctr - pred_wh * 0.5
    pred_x2y2 = pred_ctr + pred_wh * 0.5
    pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

    return pred_box

def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def preinfer(image, image_size):
    h, w, _ = image.shape
    ratio = min(image_size/h, image_size/w)
    new_h, new_w = int(h*ratio), int(w*ratio)
    top = int((image_size - new_h)/2)
    left = int((image_size - new_w)/2)
    
    image_pil = (Image.fromarray(image)).resize((new_w, new_h))
    output = np.full((image_size, image_size, 3), (114, 114, 114), dtype=np.uint8)
    output[top:top + new_h, left:left + new_w] = np.array(image_pil)
    
    output = output[:,:, ::-1].transpose([2, 0, 1]).astype(np.float32)
    output /= 255.
    output = np.expand_dims(output, 0)

    return  image, output, ratio, left, top

def postinfer(input, ratio, left, top, confidence, iou_thresh):
    input = np.transpose(input, (0, 2, 1))  # 交换维度
    input = np.squeeze(input, axis=0)

    keep = np.amax(input[:, 4:6], axis=1) > confidence    
    input = input[keep]

    bboxes = xywh2xyxy(input[:, :4])

    scores = np.max(input[:, 4:6], axis=-1, keepdims=True)  # 获取最大值
    labels = np.argmax(input[:, 4:6], axis=-1, keepdims=True) 
    
    scores = scores.flatten()  # 转换 scores 为 NumPy
    labels = labels.flatten()
    
    keep = np.zeros(len(bboxes), dtype=np.int32)
    inds = nms(bboxes, scores, iou_thresh)
    keep[inds] = 1
    keep = np.where(keep > 0)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - left
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - top
    bboxes[:, :4] /= ratio

    return labels, scores, bboxes

def infer(args, image):
    session = setup_inference(args)
    image, infer_input, ratio, left, top = preinfer(image, args.image_size)
    postinfer_input = session.run(['output'], {'input': infer_input})[0]
    labels, scores, bboxes = postinfer(postinfer_input, ratio, left, top, 
                                       confidence=0.3, iou_thresh=0.5)

    return labels, scores, bboxes

if __name__ == "__main__":
    from PIL import Image
    from io import BytesIO
    args = {
        'cuda': True,
        'det_onnx': 'platedet.onnx',
        'image_size': 640,
        'confidence': 0.5,
        'nms_thresh': 0.1,
        'class_names': ['plate', 'plate_d']
    }

    with open('imgs/沪A1707领.jpg', 'rb') as image_file:
        file_bytes = image_file.read()
    image = np.array(Image.open(BytesIO(file_bytes)).convert("RGB"))

    labels, _, bboxes = infer(args, image)

    result = []
    for index, bbox in enumerate(bboxes):
        result.append(
            [
                args['class_names'][labels[index]], 
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3])
            ]
        )

    print(result)