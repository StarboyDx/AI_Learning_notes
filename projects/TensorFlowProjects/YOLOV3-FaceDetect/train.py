import os
import shutil
import numpy as np
import tensorflow as tf
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

trainset = Dataset('train') # 定义数据集
valset = Dataset('val')
logdir = "./data/log"
steps_per_epoch = len(trainset) # 训练批数
steps_val = len(valset) # 验证数据批数
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) #全 局步数
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch # 学习率预热步数
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3]) # 占位符
conv_tensors = YOLOv3(input_tensor) # 实例化网络

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    # 解码 负责将模型输出的局部做表转换成全局坐标 进而可以去和真实的全局坐标进行损失计算
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)
model = tf.keras.Model(input_tensor, output_tensors) # 编译model
model.load_weights("./checkpoints/yolov3") # 加载权重

optimizer = tf.keras.optimizers.Adam() # 定义优化器
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir) # 定义writer

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True) # 计算预测值
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i) # 计算损失
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss # 各损失相加

        gradients = tape.gradient(total_loss, model.trainable_variables) # 计算梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # 更新参数
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


def val_step(image_data, target):
    pred_result = model(image_data, training=False) # 计算输出值
    giou_loss=conf_loss=prob_loss=0

    # optimizing process
    for i in range(3):
        conv, pred = pred_result[i*2], pred_result[i*2+1]
        loss_items = compute_loss(pred, conv, *target[i], i) # 计算损失
        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]

    return giou_loss, conf_loss, prob_loss

for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset: # 遍历训练集进行训练
        train_step(image_data, target)
    model.save_weights("./yolov3") # 保存模型
    giou_losses=conf_losses=prob_losses =0
    for image_data, target in valset: # 验证集验证
        giou_loss, conf_loss, prob_loss=val_step(image_data, target)
        giou_losses+=giou_loss/steps_val
        conf_losses+=conf_loss/steps_val
        prob_losses+=prob_loss/steps_val
    tf.print("=> VAL   giou_loss: %4.2f   conf_loss: %4.2f   "
             "prob_loss: %4.2f   total_loss: %4.2f" % (giou_loss, conf_loss,prob_loss, giou_loss+conf_loss+prob_loss))