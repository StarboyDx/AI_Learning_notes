# Notes
- 01-02 The random forest algorithm belongs to Bagging; from 03 onwards, it is Boosting.
- Xgboost融合了AdaBoost的
总结每个算法模型的调参，重要参数的意义
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds': 50,
    'eval_metric': 'auc',
    'gamma': 0,
    'max_depth': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'eta': 0.02,
    'seed': 123456,
    'nthread': 3,
    'silent': 0
}
比如这样的参数列表，结合自动化调参GridSearchCV时怎么设置，也使用这样的参数列表列举出来，
每个模型针对不同任务的一些参数区别也简单讲解区分。

算法名称、算法的一些参数、评估指标经常忘记，尤其是一些缩写，第一时间回忆不起来，导致难立刻激活记忆，串联起知识，以学习层次和顺序，总结一下涉及到的这些缩写的意思。

# Related Projects
- path: projects/KaggleProjects/01-05
        current/09 (easy)
- tips：04-过滤垃圾邮件，重点是里面函数的编写
        05-房价预测的模型，有对数据分析用到那些包的方法补充，算是对到该部分为止的一个总结项目, 有很多复杂的数据处理，很难***复习可以重点看