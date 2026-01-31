## Chapter
- 01-04: Build the First Classification Model
- 05-11: Evaluation and Validation of Classification Models
- 12-14：Logistic regression algorithm (an important basic algorithm). 
         What is more focused on here are some engineering operations, and the principles of algorithms are recorded separately.
- 15-17：K-Nearest Neighbors Algorithm (Code Examples and Parameter Tuning)
- 18：Decision tree algorithm
- 19：Support Vector Machine algorithm SVM(Widely used in text classification and multi-classification tasks)
- 20：Naive Bayes algorithm
- 21：Python operation of multiple linear regression (it can also be analyzed in Excel)
- 22: Nonlinear regression (mainly a polynomial regression is done here)
- 23：Common evaluation metrics for regression models
- 24: Regression tree（The splitting index is variance/MSE.）
- 25: Simple linear regression
- 26：Cluster analysis

# Notes
- 本章使用了很多基本的算法，复习需注意：
    根据这些代码示例去了解这些算法的重要参数和调整优化的方法
    结合原理的记录深入了解算法的来龙去脉
    注意区分不同算法对训练和测试数据的要求，文件里也有针对每种算法，对数据集做了相应的特征工程   
    关注每种算法的应用区别（各自的优缺点）加深理解

- 关于精准率和召回率：精准率和召回率关注的是模型对“正例”的判断质量，其中召回率衡量正例是否被充分找出，精准率衡量被预测为正的样本中有多少是真的；在类别不平衡或正例代价更高的任务中，Accuracy 往往失效，因此需要以业务关注的正例为核心进行评估。

09有个复杂的函数可以再分析一下