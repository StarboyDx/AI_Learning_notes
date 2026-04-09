class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {#数据文件路径
                "vocab_path": "Dataset/cnews.vocab.txt",#词表
                "train": "Dataset/cnews.train.txt",#训练集
                "val": "Dataset/cnews.val.txt",#验证集
                "test": "Dataset/cnews.test.txt"#测试集
            },
            "training_rule": {
                "embedding_dim": 256,#词向量嵌入维度
                "seq_length": 600,#文本序列最大长度
                "num_classes": 10,#分类数

                "conv1_num_filters": 128,#第一层卷积核个数
                "conv1_kernel_size": 3,#第一层卷积核尺寸

                "conv2_num_filters": 64,#第二层卷积核个数
                "conv2_kernel_size": 3,#第二层卷积核尺寸
                "hidden_dim": 128,#全连接层神经元个数

                "dropout_keep_prob": 0.5,#dropout rate
                "learning_rate": 1e-3,#学习率

                "batch_size": 64,#批次大小
                "epochs": 5,#巡礼那轮次
            },
            "result": {#模型保存路径
                "CNN_model_path": "model_storage/CNN_model.h5",#CNN模型保存路径
                "LSTM_model_path": "model_storage/LSTM_model.h5"#LSTM模型保存路径
            },

            "category":{#新闻类别
                "category":["体育", "财经", "房产",
                            "家居", "教育", "科技",
                            "时尚", "时政", "游戏", "娱乐"],
                "cat2id":{"体育":0, "财经":1, "房产":2,
                          "家居":3, "教育":4, "科技":5,
                          "时尚":6, "时政":7, "游戏":8, "娱乐":9}
            },
            "log":{#tensorboard 日志保存路径
                "CNN":"log/CNN/",
                "LSTM":"log/LSTM/"
            }
        }

    def get(self, section, name):#提取信息函数
        return self.config_dict[section][name]