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
## 四. 推导（仅供参考）
## 线性模型的推导(参考自西瓜书)

原问题：假设有$m$个样本$D=\left\{(\mathbf{x}_{1},y_{1}),(\mathbf{x}_{2},y_{2}),...,(\mathbf{x}_{m},y_{m}) \right\}$，每个样本$\mathbf{x}_{i}=\left ( x_{1},x_{2},...,x_{d} \right )$有$d$个特征，一个目标值$y_{i}=y$

### 单变量的线性回归

考虑最简单的只有**一个特征**的样本$(x_{i}, y_{i})$，线性回归试图学习：
$$
f(x_{i})=wx_{i}+b,使得f(x_{i})渐进等于y_{i}
$$
为了求得$w$和$b$，则需使用均方误差作为性能度量，并使均方误差最小化：
$$
(w, b) = \arg\limits_{(w, b)}\min\sum_{i=1}^{m}\left(wx_{i}+b-y_{i} \right)^{2}
$$
令$E(w,b)=\sum_{i=1}^{m}\left(wx_{i}+b-y_{i} \right)^{2}$分别对$w$和$b$求偏导：
$$
\begin{aligned}
\frac{\partial E(w,b)}{\partial w}&=2\cdot x_{i}\cdot \sum_{i=1}^{m}(wx_{i}+b-y_{i})
\\&=2\sum_{i=1}^{m}(wx_{i}^{2}+bx_{i}-x_{i}y_{i})
\\&=2(\sum_{i=1}^{m}wx_{i}^{2}+\sum_{i=1}^{m}x_{i}(b-y_{i})) \tag{1}
\end{aligned}
$$
对$b$:
$$
\begin{aligned}
\frac{\partial E(w,b)}{\partial b}&=2\cdot \sum_{i=1}^{m}(wx_{i}+b-y_{i})
\\&=2(\sum_{i=1}^{m}wx_{i}+\sum_{i=1}^{m}b-\sum_{i=1}^{m}y_{i})
\\&=2(\sum_{i=1}^{m}wx_{i}+mb-\sum_{i=1}^{m}y_{i})\tag{2}
\end{aligned}
$$
从以上$w$和$b$的导函数中可以看出相关变量的系数$\sum_{i=1}^{m}x_{i}^{2}$与$m$都为**正数**，故对应的导函数都为**增函数**，故当导函数值取0时对应的极值为。令$(1)$式和$(2)$式为值为0，可推导出$w$和$b$的表达式。$先令x的均值\bar{x}=\frac{1}{m}\sum_{i=1}^{m}x_{i}$，先推导$b$的表达式：
$$
令\frac{\partial E(w,b)}{\partial b}=0 \\
\Rightarrow \sum_{i=1}^{m}wx_{i}+mb-\sum_{i=1}^{m}y_{i}=0 \\
\Rightarrow b=\frac{1}{m}\sum_{i=1}^{m}y_{i}-\frac{1}{m}\sum_{i=1}^{m}wx_{i} \tag{3}
$$
再推导$w$的表达式：
$$
令\frac{\partial E(w,b)}{\partial w}=0 \\
\Rightarrow \sum_{i}^{m}wx_{i}^{2}+\sum_{i=1}^{m}x_{i}b-\sum_{i=1}^{m}x_{i}y_{i}=0
\\ 带入(3)\Rightarrow \sum_{i=1}^{m}wx_{i}^{2}+\sum_{i=1}^{m}x_{i}(\frac{1}{m}\sum_{i=1}^{m}y_{i}-\frac{1}{m}\sum_{i=1}^{m}wx_{i})-\sum_{i=1}^{m}x_{i}y_{i}=0
\\ 化简 \Rightarrow \sum_{i=1}^{m}wx_{i}^{2}+\frac{1}{m}\sum_{i=1}^{m}x_{i}\sum_{i=1}^{m}y_{i}-\frac{w}{m}(\sum_{i=1}^{m}x_{i})^{2}-\sum_{i=1}^{m}x_{i}y_{i}=0
\\ 化简合并 \Rightarrow w(\sum_{i=1}^{m}x_{i}^2-\frac{1}{m}(\sum_{i=1}^{m}x_{i})^{2})+\sum_{i=1}^{m}(\bar{x}-x_{i})y_{i}=0
\\ \Rightarrow w = \frac{\sum_{i=1}^{m}y_{i}(x_{i}-\bar{x})}{\sum_{i=1}^{m}x_{i}^2-\frac{1}{m}(\sum_{i=1}^{m}x_{i})^{2}}
$$
求得的两个表达式为：
$$
w = \frac{\sum_{i=1}^{m}y_{i}(x_{i}-\bar{x})}{\sum_{i=1}^{m}x_{i}^2-\frac{1}{m}(\sum_{i=1}^{m}x_{i})^{2}}, \qquad
b=\frac{1}{m}\sum_{i=1}^{m}y_{i}-\frac{1}{m}\sum_{i=1}^{m}wx_{i}
$$

### 多元线性回归

考虑多个特征的样本$\mathbf{x}_{i}=\left ( x_{1},x_{2},...,x_{d} \right )$，多元线性回归试图学习：
$$
f(x_{i})=w^{T}x_{i}+b,使得f(x_{i})渐进等于y_{i}
$$
为了便于向量的运算，令$\hat{w}=(w;b)=\binom{w}{b}$，把数据集$D$表示为一个$m\times(d+1)大小的矩阵：$
$$
X=\begin{Bmatrix}
 &x_{11}  &x_{12}&...&x_{1d} &1\\
 &x_{21}  &x_{22}&...&x_{2d} &1\\
 &...  &...  &...  &... &...\\
 &x_{m1}  &x_{m2}  &...  &x_{md} &1
\end{Bmatrix}=\begin{pmatrix}
\mathbf{x}_{1}^{T} &1\\
\mathbf{x}_{2}^{T} &1\\
... &...\\
\mathbf{x}_{m}^{T} &1
\end{pmatrix}
$$
再把目标值也写成向量的形式：$\mathbf{y}=\begin{pmatrix}
y_{1}\\
y_{2}\\
...\\
y_{m}
\end{pmatrix}$，则类似有：
$$
\hat{w}=\arg\min(\mathbf{y}-X\hat{w})^{T}(\mathbf{y}-X\hat{w})
$$
令$E_{\hat{w}}=(\mathbf{y}-X\hat{w})^{T}(\mathbf{y}-X\hat{w})$,对$\hat{w}$求导得：
$$
\begin{aligned}
\frac{\partial E_{\hat{w}}}{\partial \hat{w}}&=\triangledown_{\hat{w}}(\mathbf{y}^{T}-\hat{w}^{T}X^{T})(\mathbf{y}-X\hat{w})
\\ &=\triangledown_{\hat{w}}(\mathbf{y}^{T}\mathbf{y}-y^{T}X\hat{w}-\hat{w}^{T}X^{T}\mathbf{y}+\hat{w}^{T}X^{T}X\hat{w})
\\ &=0-(\mathbf{y}^{T}X)^{T}-X^{T}\mathbf{y}+(X^{T}X\hat{w}+(\hat{w}X^{T}X)^{T})
\\ &=-2X^{T}\mathbf{y}+2X^{T}X\hat{w}
\\ &=2X^{T}(X\hat{w}-\mathbf{y})
\end{aligned}
\\ 这里的矩阵求导法则用了如下两个公式: \\ 
\frac{\partial(A^{T}WB)}{\partial W}=AB^{T},\frac{\partial(A^{T}W^{T}B)}{\partial W}=BA^{T}
$$
令上式为零即可推导出$\hat{w}$的表达式：
$$
2X^{T}(X\hat{w}-\mathbf{y})=0
\Rightarrow X^{T}X\hat{w}=X^{T}\mathbf{y}
\\ \Rightarrow \hat{w}=(X^{T}X)^{-1}X^{T}\mathbf{y}
$$
令每个样本为$\hat{x}_{i}=(\mathbf{x}_{i},1)$，得最终的多元线性回归模型：
$$
\begin{aligned}
f(\hat{x}_{i})&=(\mathbf{x}_{i},1)\begin{pmatrix}
w\\
b
\end{pmatrix}=\hat{x}_{i}\hat{w}\\&=\hat{x}_{i}(X^{T}X)^{-1}X^{T}\mathbf{y}
\end{aligned}
$$
显然，上式能成立的基本条件就是矩阵$X^{T}X$可逆（即满秩矩阵），**当特征的数量大于样本的数量时矩阵就不可逆了**。

### 广义线性模型

$$
g(y)=w^{T}x+b\\ \Rightarrow \\
y = g^{-1}(w^{T}x+b)
$$

*作者才疏学浅，以上资料仅供参考，交流。*
