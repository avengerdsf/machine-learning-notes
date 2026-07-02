# 模型评估

训练模型的目标不是让模型记住训练数据，而是让模型对没有见过的新数据也能给出准确预测。训练误差只能说明模型对训练集的拟合程度，不能单独衡量模型的泛化能力，因此需要保留一部分不参与训练的数据来评估模型。

## 1. 划分训练集和测试集

设数据集包含 $m$ 个样本：

$$
\left\{
\left(\mathbf{x}^{(1)},y^{(1)}\right),
\left(\mathbf{x}^{(2)},y^{(2)}\right),
\ldots,
\left(\mathbf{x}^{(m)},y^{(m)}\right)
\right\}
$$

课程中先将数据随机划分为训练集和测试集：

$$
70\% \text{ training set},
\qquad
30\% \text{ test set}
$$

训练集用于学习模型参数 $\mathbf{w}$ 和 $b$，测试集不参与参数学习，只用于评估训练完成后的模型在新数据上的表现。划分前通常需要打乱样本，避免原始数据顺序使两个集合的数据分布不同。

用 $m_{\text{train}}$ 和 $m_{\text{test}}$ 分别表示训练集和测试集的样本数量：

$$
m=m_{\text{train}}+m_{\text{test}}
$$

## 2. 回归模型的评估

以回归模型为例，先在训练集上最小化带正则化的代价函数：

$$
J(\mathbf{w},b)
=
\frac{1}{2m_{\text{train}}}
\sum_{i=1}^{m_{\text{train}}}
\left(
f_{\mathbf{w},b}\left(\mathbf{x}^{(i)}\right)-y^{(i)}
\right)^2
+
\frac{\lambda}{2m_{\text{train}}}
\sum_{j=1}^{n}w_j^2
$$

训练完成后，分别计算训练误差和测试误差：

$$
J_{\text{train}}(\mathbf{w},b)
=
\frac{1}{2m_{\text{train}}}
\sum_{i=1}^{m_{\text{train}}}
\left(
f_{\mathbf{w},b}\left(\mathbf{x}_{\text{train}}^{(i)}\right)
-y_{\text{train}}^{(i)}
\right)^2
$$

$$
J_{\text{test}}(\mathbf{w},b)
=
\frac{1}{2m_{\text{test}}}
\sum_{i=1}^{m_{\text{test}}}
\left(
f_{\mathbf{w},b}\left(\mathbf{x}_{\text{test}}^{(i)}\right)
-y_{\text{test}}^{(i)}
\right)^2
$$

$J_{\text{train}}$ 和 $J_{\text{test}}$ 都不包含正则化项。正则化项用于训练时约束参数大小，不是模型预测误差的一部分；评估时需要直接比较预测值和真实值之间的误差。

如果 $J_{\text{train}}$ 很小，而 $J_{\text{test}}$ 明显更大，说明模型虽然很好地拟合了训练数据，但没有很好地泛化到新数据。

## 3. 分类模型的评估

对于二分类模型，先把模型输出转换为预测类别：

$$
\hat{y}^{(i)}
=
\begin{cases}
1, & f_{\mathbf{w},b}\left(\mathbf{x}^{(i)}\right)\ge 0.5\\
0, & f_{\mathbf{w},b}\left(\mathbf{x}^{(i)}\right)<0.5
\end{cases}
$$

分类误差是预测错误的样本比例。定义指示函数：

$$
\mathbb{1}\left\{\hat{y}^{(i)}\ne y^{(i)}\right\}
=
\begin{cases}
1, & \hat{y}^{(i)}\ne y^{(i)}\\
0, & \hat{y}^{(i)}=y^{(i)}
\end{cases}
$$

训练误差和测试误差分别为：

$$
J_{\text{train}}(\mathbf{w},b)
=
\frac{1}{m_{\text{train}}}
\sum_{i=1}^{m_{\text{train}}}
\mathbb{1}
\left\{
\hat{y}_{\text{train}}^{(i)}
\ne
y_{\text{train}}^{(i)}
\right\}
$$

$$
J_{\text{test}}(\mathbf{w},b)
=
\frac{1}{m_{\text{test}}}
\sum_{i=1}^{m_{\text{test}}}
\mathbb{1}
\left\{
\hat{y}_{\text{test}}^{(i)}
\ne
y_{\text{test}}^{(i)}
\right\}
$$

例如，测试集包含 $100$ 个样本，其中 $8$ 个样本预测错误，则测试误差为：

$$
J_{\text{test}}=\frac{8}{100}=0.08
$$

对应的测试准确率为：

$$
\text{accuracy}=1-J_{\text{test}}=0.92
$$

## 4. 模型评估的基本流程

先随机划分训练集和测试集，再只使用训练集学习参数，最后在训练集和测试集上分别计算误差。训练误差衡量模型对已见数据的拟合程度，测试误差用于估计模型对未见数据的泛化误差。

测试集必须与训练过程隔离。如果根据测试误差反复修改模型，测试集就参与了模型决策，得到的测试误差将不再是对最终模型泛化能力的独立评估。模型选择和交叉验证需要引入验证集，将在后续笔记中单独讨论。
