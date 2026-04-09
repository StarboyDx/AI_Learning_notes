import tensorflow.keras as keras #引入keras
import tensorflow as tf #引入tf
from config import Config #配置类
import os
from sklearn import metrics #模型评估  输出分类报告
import numpy as np
from tqdm import tqdm #进度条
from datetime import datetime #日期模块

class EpochLoss(tf.keras.metrics.Metric):
    """
    负责在一个轮次的迭代训练中累计平均损失
    """
    def __init__(self, name='EpochLoss', **kwargs):
        super(EpochLoss, self).__init__(name=name, **kwargs)
        #定义统计总损失值的变量
        self.total = self.add_weight(name='total', dtype=tf.float32, initializer=tf.zeros_initializer())
        #定义统计样本个数的变量
        self.count = self.add_weight(name='count', dtype=tf.float32, initializer=tf.zeros_initializer())
    def update_state(self, y_true, loss, sample_weight=None):
        values = y_true.shape[0]*loss
        self.total.assign_add(values)#累计损失值
        self.count.assign_add(y_true.shape[0])#累计样本数

    def result(self):
        return self.total / self.count#计算平均损失

    def reset_states(self):
        self.total.assign(0)#损失总和归零
        self.count.assign(0)#样本数

def train_(model, optimizer, loss_object, train_data,
           val_data, epochs, model_save_path, summary_writer):
    """

    :param model: 需要训练的模型
    :param optimizer: 优化器对象
    :param loss_object: 损失函数对象
    :param train_data: 训练集
    :param val_data: 验证集
    :param epochs: 训练轮次
    :param model_save_path: 模型保存路径
    :param summary_writer: tensorboard日志写入器
    """
    max_acc = 0
    loss_calcer = EpochLoss()  # 定义损失统计器
    acc = tf.keras.metrics.CategoricalAccuracy()  # 定义准确率统计器
    for epoch in range(1, epochs + 1):
        # 遍历数据集进行训练
        with tqdm(total=len(train_data), ncols=120) as _tqdm:
            _tqdm.set_description("epoch-train:{}/{}".format(epoch, epochs))  # 进度条前缀信息
            for x_train, y_train in train_data:  # 遍历数据
                # 开一个gradient tape, 计算梯度
                with tf.GradientTape() as tape:
                    predictions = model(x_train, training=True)  # 前向传播计算输出值
                    loss = loss_object(y_train, predictions)  # 计算损失值
                    gradients = tape.gradient(loss, model.trainable_variables)  # 反向传播计算梯度值
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 优化更新参数
                loss_calcer.update_state(y_train, loss)  # 累计平均损失
                acc.update_state(y_train, predictions)  # 累计平均acc
                train_acc = acc.result()  # 计算当前准确率
                train_loss = loss_calcer.result()  # 计算当前平均损失
                _tqdm.set_postfix(Training_loss='{:.6f}'.format(train_loss),
                                  Training_acc='{:.6f}'.format(train_acc))  # 进度条后缀显示loss和acc
                _tqdm.update(1)  # 进度条更新

        # 遍历验证机进行验证
        acc.reset_states()  # 准确率统计器归零
        loss_calcer.reset_states()  # 损失统计器归零
        # 遍历数据集进行训练
        with tqdm(total=len(val_data), ncols=120) as _tqdm:
            _tqdm.set_description("epoch-val:{}/{}".format(epoch, epochs))
            for x_val, y_val in val_data:  # 不需要求梯度
                predictions = model(x_val, training=False)  # 计算输出值
                loss = loss_object(y_val, predictions)  # 计算损失值
                loss_calcer.update_state(y_val, loss)  # 累计平均损失
                acc.update_state(y_val, predictions)  # 累计平均acc
                val_acc = acc.result()
                val_loss = loss_calcer.result()
                _tqdm.set_postfix(val_loss='{:.6f}'.format(val_loss), val_acc='{:.6f}'.format(val_acc))
                _tqdm.update(1)
        if val_acc > max_acc:  # 判断验证集指标是否提升，如果提升就保存模型
            model.save(model_save_path, overwrite=True)
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)  # 监听损失值
            tf.summary.scalar('train_acc', train_acc, step=epoch)  # 监听准确率
            tf.summary.scalar('val_loss', val_loss, step=epoch)  # 监听损失值
            tf.summary.scalar('val_acc', val_acc, step=epoch)  # 监听准确率


def test_(model, data):
    # 遍历数据集进行训练
    pred_test = []
    y_test = []
    for x, y in tqdm(data):  # 不需要求梯度
        pred = model.predict(x)  # 计算输出值
        pred_test.append(pred)  # 保存预测值
        y_test.append(y)  # 保存真实值
    pred_test = np.concatenate(pred_test, axis=0)  # 整合成一个数组
    y_test = np.concatenate(y_test, axis=0)  # 整合成一个数组
    # 打印分类报告
    print(metrics.classification_report(np.argmax(pred_test, axis=1), np.argmax(y_test, axis=1)))


class TextCNN(object):

    def __init__(self,vocab_size):
        self.config = Config()#实例化配置类
        self.vocab_size=vocab_size#词表大小 跟嵌入矩阵有关

    def model(self):
        num_classes = self.config.get("training_rule", "num_classes")#获取类别数
        embedding_dim=self.config.get("training_rule", "embedding_dim")#获取词向量维度

        conv1_num_filters = self.config.get("training_rule", "conv1_num_filters")#获取第一层卷积核个数
        conv1_kernel_size = self.config.get("training_rule", "conv1_kernel_size")#获取第一层卷积核尺寸

        conv2_num_filters = self.config.get("training_rule", "conv2_num_filters")#获取第二层卷积核个数
        conv2_kernel_size = self.config.get("training_rule", "conv2_kernel_size")#获取第二层卷积核尺寸

        hidden_dim = self.config.get("training_rule", "hidden_dim")#获取全连接层神经元个数
        dropout_keep_prob = self.config.get("training_rule", "dropout_keep_prob")

        model_input = keras.layers.Input((None,))
        embedding_layer = keras.layers.Embedding(self.vocab_size, embedding_dim)#嵌入层
        embedded = embedding_layer(model_input)

        conv_1 = keras.layers.Conv1D(conv1_num_filters, conv1_kernel_size, padding="SAME")(embedded)#卷积1
        conv_2 = keras.layers.Conv1D(conv2_num_filters, conv2_kernel_size, padding="SAME")(conv_1)#卷积2
        max_poolinged = keras.layers.GlobalMaxPool1D()(conv_2)#序列长度维度上全局池化

        full_connect = keras.layers.Dense(hidden_dim)(max_poolinged)#全连接
        droped = keras.layers.Dropout(dropout_keep_prob)(full_connect)#dropout
        relued = keras.layers.ReLU()(droped)
        model_output = keras.layers.Dense(num_classes, activation="softmax")(relued)#输出层
        model = keras.models.Model(inputs=model_input, outputs=model_output)#编译模型
        print(model.summary())#打印模型概述
        return model

    def train(self, train_data,val_data):
        log_dir = self.config.get("log", "CNN") + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(log_dir)#创建summary_writer
        model_save_path = self.config.get("result", "CNN_model_path")#获取模型保存路径
        epochs=self.config.get("training_rule", "epochs")#获取训练轮数
        learning_rate=self.config.get("training_rule", "learning_rate")#获取学习率

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)#定义优化器
        loss_obj=tf.keras.losses.CategoricalCrossentropy()#定义损失函数
        model = self.model()#实例化模型
        train_(model=model,optimizer=optimizer,
              loss_object=loss_obj,
              train_data=train_data,
              val_data=val_data,
              epochs=epochs,
              model_save_path=model_save_path,
              summary_writer=self.summary_writer)#训练

    def test(self,test_data):
        model_save_path = self.config.get("result", "CNN_model_path")#获取模型路径
        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)#加载模型
            print("-----model loaded-----")
            model.summary()
        test_(model,test_data)#模型测试


class TextLSTM(object):

    def __init__(self, vocab_size):
        self.config = Config()
        self.vocab_size = vocab_size

    def model(self):
        num_classes = self.config.get("training_rule", "num_classes")
        embedding_dim = self.config.get("training_rule", "embedding_dim")

        model_input = keras.layers.Input((None,))
        embedding = keras.layers.Embedding(self.vocab_size, embedding_dim)(model_input)
        LSTM = keras.layers.LSTM(256)(embedding)
        FC1 = keras.layers.Dense(256, activation="relu")(LSTM)
        droped = keras.layers.Dropout(0.5)(FC1)
        FC2 = keras.layers.Dense(num_classes, activation="softmax")(droped)

        model = keras.models.Model(inputs=model_input, outputs=FC2)

        model.summary()
        return model

    def train(self, train_data, val_data):
        log_dir = self.config.get("log", "LSTM") + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        model_save_path = self.config.get("result", "LSTM_model_path")
        epochs = self.config.get("training_rule", "epochs")
        learning_rate = self.config.get("training_rule", "learning_rate")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        acc = tf.keras.metrics.CategoricalAccuracy()
        loss_obj = tf.keras.losses.CategoricalCrossentropy()
        model = self.model()
        train_(model=model, optimizer=optimizer,
               loss_object=loss_obj,
               train_data=train_data,
               val_data=val_data,
               epochs=epochs,
               model_save_path=model_save_path,
               summary_writer=self.summary_writer)

    def test(self, test_data):
        model_save_path = self.config.get("result", "LSTM_model_path")

        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        test_(model, test_data)
