from config import Config #配置类
from preprocess import preprocess #文本预处理函数
import tensorflow as tf
import numpy as np

class NewsClassifier(object):
    def __init__(self, mode='CNN'):
        self.config = Config()  # 获取配置信息
        model_path = self.config.get("result", mode + "_model_path")  # 获取模型路径
        self.model = tf.keras.models.load_model(model_path)  # 加载模型
        self.id2cat = self.config.get("category", "category")  # 获取id转类别的列表

    def predict(self, sentence):
        x = preprocess(sentence)  # 对文本进行预处理
        y_pred = self.model.predict(x)  # 模型预测
        y_class_id = np.argmax(y_pred, axis=-1)  # 获取每个样本最大概率类别下标
        y_p = list(y_pred[range(y_pred.shape[0]), y_class_id])  # 获取每个样本最大概率类别概率
        y_class = [self.id2cat[i] for i in y_class_id]  # id转换成对应的类别名称
        return y_class, y_p

if __name__ == '__main__':
    sentences=[
        """黄蜂vs湖人首发：科比冲击七连胜 火箭两旧将登场新浪体育讯北京时间3月28日，
        NBA常规赛洛杉矶湖人主场迎战新奥尔良黄蜂，赛前双方也公布了首发阵容：点击进入
        新浪体育视频直播室点击进入新浪体育图文直播室点击进入新浪体育NBA专题点击进入
        新浪NBA官方微博双方首发阵容：湖人队：德里克-费舍尔、科比-布莱恩特、罗恩-阿泰
        斯特、保罗-加索尔、安德鲁-拜纳姆黄蜂队：克里斯-保罗、马科-贝里内利、特雷沃-
        阿里扎、卡尔-兰德里、埃梅卡-奥卡福(新浪体育)""",

        """PL系列高端 烟台三星PL65家用DC热卖【ZOL-七天在线 it7t.com】三星PL
        系列推出了多款新机，其中PL65是该系列中的最高端机型，其拥有1220万有效像素，
        5倍的光学变焦。还加入了场景识别的功能使用户的拍摄更加简单。近期准备购置相机
        的家庭用户可以考虑一下。""",
    ]
    clf=NewsClassifier(mode='LSTM')#实例化推理类
    print(clf.predict(sentences))#调用predict方法完成推理