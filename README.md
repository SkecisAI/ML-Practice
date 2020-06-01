# 机器学习-模型程序实现
基于周志华西瓜书中的数据集和各个机器学习模型的算法原理，采用python编写代码
## 一. 模型清单
* **逻辑回归**
* **线性判别分析**
* **决策树**  
	1. 划分选择  
		1. 基于信息增益  
    	2. 基于基尼指数  
	2. 剪枝  
		1. 预剪枝  
		2. 后剪枝  
* **神经网络**  
	1. 标准BP算法  
* **朴素贝叶斯**  
	1. 拉普拉斯变换
* **（待续）**
## 二. 代码清单
代码中重要的部分都配有蹩脚的英文注释（T.T苦学英语ing），具体使用的数据集内容请参考代码引用和数据集清单。
* [logistic_regression](https://github.com/SkecisAI/ML-Practice/blob/master/logistic_regression.py): 逻辑回归实现。
* [linear_discriminant_analysis](https://github.com/SkecisAI/ML-Practice/blob/master/linear_discriminant_analysis.py): 线性判别分析实现。
* [decision_tree](https://github.com/SkecisAI/ML-Practice/blob/master/decision_tree.py): 决策树实现，集成各个思想，包含划分方式有`信息增益`、`基尼指数`，连续值的处理方式有`平均值法`、`二分法`，
				 以及剪枝方法有`预剪枝`、`后剪枝`。
* [neural_network](https://github.com/SkecisAI/ML-Practice/blob/master/neural_network.py): 神经网络实现，标准BP算法。
* [naive_bayes_classifier](https://github.com/SkecisAI/ML-Practice/blob/master/naive_bayes_classifier.py): 朴素贝叶斯实现，采用了拉普拉斯变换。
* （待续）
## 三. 数据集清单
均为csv文件，内容如下
* [西瓜数据集3.0alpha](https://github.com/SkecisAI/ML-Practice/blob/master/watermelon.csv)
* [西瓜数据集2.0](https://github.com/SkecisAI/ML-Practice/blob/master/watermelon4.csv)  
* [西瓜数据集3.0](https://github.com/SkecisAI/ML-Practice/blob/master/watermelon3.csv)

*作者才疏学浅，以上资料仅供参考，交流。*
