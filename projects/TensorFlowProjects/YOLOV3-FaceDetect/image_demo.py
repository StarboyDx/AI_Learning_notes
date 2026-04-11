import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
import matplotlib.pyplot as plt
from core.config import cfg

input_size   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
image_path   = "./docs/demo2.jpg"
original_image      = cv2.imread(image_path)#读取原始图片 用于绘图
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])#预处理图片
image_data = image_data[np.newaxis, ...].astype(np.float32)

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])#输入tensor
feature_maps = YOLOv3(input_layer)#构建模型
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)#解码
    bbox_tensors.append(tf.reshape(bbox_tensor, (1,-1, 5+NUM_CLASS)))
bbox_tensors = tf.concat(bbox_tensors, axis=1)
model = tf.keras.Model(input_layer, bbox_tensors)#编译模型
model.load_weights("./checkpoints/yolov3")#加载权重
model.summary()#模型描述

pred_bbox = model.predict(image_data).reshape(-1,6)#模型预测
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)#将预测框转换到缩放前的大小
bboxes = utils.nms(bboxes, 0.45, method='nms')#非极大值抑制
image = utils.draw_bbox(original_image, bboxes)#绘制预测框
plt.figure(figsize=(20,10))
plt.imshow(image)#展示结果图片
# plt.show()
# Update: Docker 无图形界面，将 plt.show() 替换为保存图片到本地
plt.savefig("result_demo.jpg", bbox_inches='tight', pad_inches=0.0)
print("=> 预测完成！图片已保存为 result_demo.jpg")