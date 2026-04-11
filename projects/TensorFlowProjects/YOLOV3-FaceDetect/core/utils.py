import cv2
import random
import colorsys
import numpy as np
from core.config import cfg

#读取类别列表
def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n') # 读取类别名称
    return names

# 读取anchor尺寸文件
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline() # 读取尺寸
    anchors = np.array(anchors.split(','), dtype=np.float32) # 转换成float数组
    return anchors.reshape(3, 3, 2)

# 训练图像预处理
def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size # 目标尺寸
    h, w, _ = image.shape # 原尺寸

    scale = min(iw/w, ih/h) # 选择宽度和高度相对原图缩放比例最小的比例进行缩放，保证原图信息都在
    nw, nh = int(scale * w), int(scale * h) # 计算缩放后的w和h
    image_resized = cv2.resize(image, (nw, nh)) # 进行等比例缩放, 相对目标尺寸可能会有空白

    image_padded = np.full(shape = [ih, iw, 3], fill_value = 128.0) # 目标尺寸填充空白
    dw, dh = (iw - nw) // 2, (ih - nh) // 2 # 计算填充下标
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized # 将缩放后的原图居中放置
    image_padded = image_padded / 255. # 归一化到0-1

    if gt_boxes is None:
        return image_padded
    
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw # 对原来的真实框坐标进行对应的缩放
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes
    
# 绘制人脸检测结果
def draw_bbox(image, bboxes, classes = read_class_names(cfg.YOLO.CLASSES), show_label = True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape # 图像尺寸
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) # 每一类不同颜色
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes): # 遍历预测框
        coor = np.array(bbox[:4], dtype=np.int32) # 获取坐标
        fontScale = 0.5
        score = bbox[4] # 获取置信度
        class_ind = int(bbox[5]) # 获取类别下标
        bbox_color = colors[class_ind] # 获取对应类别颜色
        bbox_thick = int(0.6 * (image_h + image_w) / 600) # 计算框线线粗值
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick) # 绘制矩形框

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score) # 打印类别和置信度
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0] # 计算字号大小
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # 绘制信息矩阵框
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA) # 绘制文字信息
    return image

# 计算两个框的iou
def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    # 注意 ... 这里是一批框
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1]) # boxes1面积
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) # boxes2面积

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2]) # 重叠区域左上角
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:]) # 重叠区域右下角作保

    inter_section = np.maximum(right_down - left_up, 0.0) # 判断是否重叠，比0小就不重叠，设置为0
    inter_area = inter_section[..., 0] * inter_section[..., 1] # 重叠部分面积
    union_area = boxes1_area + boxes2_area - inter_area # 并集面积
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps) # 交并集

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5])) # 获取预测框中的所有类别
    best_bboxes = []

    for cls in classes_in_img: # 遍历每一个类别
        cls_mask = (bboxes[:, 5] == cls) # 筛选出当前类别的框
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4]) # 提取分值最大的框下标
            best_bbox = cls_bboxes[max_ind] # 提取分值最大的框
            best_bboxes.append(best_bbox) # 放入保留列表
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]]) # 过滤掉best_box
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4]) # 计算best_box和其他box的iou
            weight = np.ones((len(iou),), dtype=np.float32) # 初始权重都为1

            assert method in ['nms', 'soft-nms']

            if method == 'nms': 
                iou_mask = iou > iou_threshold # 判读iou'是否超过阈值
                weight[iou_mask] = 0.0 # 超过阈值的框权重设为0

            if method == 'soft-nms': 
                weight = np.exp(-(1.0 * iou ** 2 / sigma)) # 根据iou减小置信度
            # 根据权重对框进行筛选，小于等于0的丢掉
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

#将预测框转换到图片缩放前的尺寸
def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]#合法缩放比例范围
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]#预测狂坐标
    pred_conf = pred_bbox[:, 4]#预测框置信度
    pred_prob = pred_bbox[:, 5:]#预测狂概率

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax) 坐标转换
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)#缩放比例

    dw = (input_size - resize_ratio * org_w) / 2#偏移量
    dh = (input_size - resize_ratio * org_h) / 2

    #反向缩放和平移 计算出预测狂在缩放前的原图上的坐标
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio#计算出对应图片缩放前的框坐标
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range 超出部分裁剪
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes去掉一些非法框
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores去掉置信度分数低的框
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)