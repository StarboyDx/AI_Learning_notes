import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg

#定义数据集
class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        #获取相关参数
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150
        #读取标注文件数据
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines() # 读取所有行
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]#按照空格拆分
        np.random.shuffle(annotations) # 打乱数据
        return annotations
    
    # 随机翻转
    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :] # 图像翻转
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]] # 真实框对应翻转

        return image, bboxes
    
        #随机裁剪
    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            #提取所有框的最大外接矩形
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            #获取裁剪范围 防止将真实框裁剪掉
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            #随机数生成裁剪坐标
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            #裁剪
            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            #真实框对应的进行平移
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
    
    # 随机平移
    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            # 根据最小左上角和最大右下角获得所有框的最大外界矩形
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # 获取平移的范围
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            # 随机数生成平移向量  范围控制以防将真实框平移出边界
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            # 图像进行平移
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            # 真实框坐标对应平移
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        
        return image, bboxes
    
    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = os.getcwd()+line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)#读取图像
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]]) # 获取真实框坐标（x1,y1,x2,y2）
        #数据增强
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes)) #翻转
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes)) #裁剪
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes)) #偏移

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#转换颜色模式
        #缩放填充图片 并对真实框坐标进行对应的缩放
        image, bboxes = utils.image_preprocess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes
    
    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        #坐标转换 (x,y,w,h)->(x1,y1,x2,y2)
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        return utils.bboxes_iou(boxes1,boxes2)
    
    # 对真实框进行编码得到训练标签
    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)] # 定义类别label数组
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)] # 定义坐标label数组
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4] # 获取真实框坐标
            bbox_class_ind = bbox[4] # 获取真实标签

            onehot = np.zeros(self.num_classes, dtype=np.float64) # 类别标签onehot处理
            onehot[bbox_class_ind] = 1.0 # 注入真实标签
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution # label_smooth
            # 根据特征图坐标网格得到锚框anchors的坐标
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis] # 使用不同步长进行缩小

            iou = []
            exist_positive = False
            for i in range(3): # 遍历计算3个不同尺度特征图的先验框
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) # 定义数组存储三种不同的anchor中心点和高宽
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5 # 中心点坐标
                anchors_xywh[:, 2:4] = self.anchors[i] # anchor框尺寸

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh) # 计算每一个真实框和anchors的iou
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3 # 剔除iou不足0.3的配对

                if np.any(iou_mask): # 如果有>0.3的
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive: #如果没有匹配的anchor
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1) # 取出最大的anchor id
                best_detect = int(best_anchor_ind / self.anchor_per_scale) # 除以3  三个尺寸的检测器
                best_anchor = int(best_anchor_ind % self.anchor_per_scale) # 再除以3 每个尺寸的检测器对应3个anchor
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32) # 提取坐标
                # 注入标签
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)#随机选择训练尺寸
            self.train_output_sizes = self.train_input_size // self.strides#计算输出尺寸
            #定义批次数据tensor
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)
            #定义小尺寸anchor层标签tensor
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            #定义中尺寸anchor层标签tensor
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            #定义大尺寸anchor层标签tensor
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num#计算批次数据截取起始下标
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]#获取标注信息
                    image, bboxes = self.parse_annotation(annotation)#解析标注
                    #根据真实框制作训练标签
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    #注入标签
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)#打乱数据
                raise StopIteration

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.num_batchs