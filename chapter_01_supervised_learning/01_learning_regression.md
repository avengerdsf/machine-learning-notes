# 线性回归模型

线性回归是监督学习中的基础回归算法，用输入特征预测连续数值输出。参考吴恩达《Machine Learning Specialization》中对线性回归的讲解，学习线性回归时需要把问题拆成四件事：模型表示、代价函数、参数学习和训练诊断。

## 1. 回归问题

回归任务的目标是学习输入 $x$ 到连续输出 $y$ 的映射关系，例如：

- 根据房屋面积预测房价
- 根据学习时长预测考试成绩
- 根据历史销量预测未来销量

训练集通常表示为：

$$
\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})\}
$$

其中，$m$ 表示样本数量，$x^{(i)}$ 表示第 $i$ 个样本的输入特征，$y^{(i)}$ 表示第 $i$ 个样本的真实输出。

## 2. 模型形式

### 2.1 单变量线性回归

单变量线性回归只有一个输入特征，模型假设函数为：

$$
f_{w,b}(x) = wx + b
$$

其中，$w$ 是权重参数，$b$ 是偏置参数，$f_{w,b}(x)$ 是模型预测值。

### 2.2 多变量线性回归

多变量线性回归包含多个输入特征，模型假设函数为：

$$
f_{\mathbf{w},b}(\mathbf{x}) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

向量化写法为：

$$
f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b
$$

其中，$\mathbf{x}$ 是特征向量，$\mathbf{w}$ 是权重向量，$n$ 是特征数量。

## 3. 代价函数

**线性回归常用平方误差代价函数衡量预测值与真实值之间的差距。**原始均方误差通常写作 $\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2$；课程中写成 $\frac{1}{2m}$ 是为了让平方项求导后产生的 $2$ 与分母中的 $2$ 抵消，使梯度公式更简洁，不改变最小值对应的参数。

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2
$$


多变量形式为：

$$
J(\mathbf{w},b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2
$$

训练目标是**找到使 $J$ 最小的参数 $\mathbf{w}$ 和 $b$。**

## 4. 梯度下降

梯度下降通过反复更新参数来最小化代价函数：

$$
w = w - \alpha \frac{\partial J(w,b)}{\partial w}
$$

$$
b = b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

其中，$\alpha$ 是学习率，控制每次参数更新的步长。

单变量线性回归中，梯度为：

$$
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
$$

$$
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})
$$

参数需要同步更新，即先用当前 $w$ 和 $b$ 计算两个梯度，再同时更新 $w$ 和 $b$。

## 5. 学习率

学习率 $\alpha$ 直接影响训练过程：

- $\alpha$ 太小：代价函数下降慢，训练时间长
- $\alpha$ 合理：代价函数稳定下降
- $\alpha$ 太大：参数更新越过最小值，代价函数可能上升或发散

训练时可以记录每轮迭代的 $J$ 值。如果梯度下降实现正确，并且学习率合适，$J$ 应随迭代次数整体下降。

## 6. 特征缩放

当不同特征的取值范围差异很大时，梯度下降会收敛得更慢。特征缩放可以把不同特征调整到相近范围。

常用标准化方式为：

$$
x_j = \frac{x_j - \mu_j}{\sigma_j}
$$

其中，$\mu_j$ 是第 $j$ 个特征的均值，$\sigma_j$ 是第 $j$ 个特征的标准差。

示例：

```python
import numpy as np

X = np.array([[2104, 5],
              [1416, 3],
              [1534, 3]])

mu = X.mean(axis=0)
sigma = X.std(axis=0)
X_scaled = (X - mu) / sigma
```

## 7. 多项式回归

线性回归的“线性”指模型对参数 $\mathbf{w}$ 和 $b$ 是线性的，不限制原始特征只能以一次项出现。对于非线性趋势，可以通过构造新特征扩展模型表达能力。

例如，用房屋面积 $x$ 预测价格时，可以构造：

$$
f_{w,b}(x) = w_1x + w_2x^2 + w_3x^3 + b
$$

此时模型仍然可以按线性回归方式训练，因为 $x$、$x^2$、$x^3$ 被视为三个输入特征。

## 8. 训练流程

线性回归的基本训练流程如下：

1. 准备训练数据 $X$ 和 $y$
2. 初始化参数 $\mathbf{w}$ 和 $b$
3. 计算预测值 $f_{\mathbf{w},b}(\mathbf{x})$
4. 计算代价函数 $J(\mathbf{w},b)$
5. 计算梯度
6. 使用梯度下降更新参数
7. 重复训练，直到代价函数收敛

## 9. NumPy 实现示例

```python
import numpy as np


def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    errors = X @ w + b - y
    dj_dw = X.T @ errors / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters):
    cost_history = []

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


def z_score_normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def predict(X, w, b):
    return X @ w + b


if __name__ == "__main__":
    X_train = np.array([
        [2104, 5],
        [1416, 3],
        [1534, 3],
        [852, 2],
        [1940, 4],
    ], dtype=float)
    y_train = np.array([460, 232, 315, 178, 389], dtype=float)

    X_norm, mu, sigma = z_score_normalize(X_train)

    w_init = np.zeros(X_norm.shape[1])
    b_init = 0.0
    alpha = 0.1
    num_iters = 1000

    w, b, cost_history = gradient_descent(
        X_norm,
        y_train,
        w_init,
        b_init,
        alpha,
        num_iters,
    )

    house = np.array([[1600, 3]], dtype=float)
    house_norm = (house - mu) / sigma
    price = predict(house_norm, w, b)

    print("初始代价:", round(cost_history[0], 4))
    print("最终代价:", round(cost_history[-1], 4))
    print("学习到的 w:", np.round(w, 4))
    print("学习到的 b:", round(b, 4))
    print("预测房价:", round(price[0], 2))
```

## 11. 小结

线性回归的核心不是公式本身，而是完整建模思路：用假设函数表示预测关系，用代价函数衡量误差，用梯度下降最小化代价函数，再通过学习率、特征缩放和训练曲线判断训练是否有效。

## 参考资料

- Andrew Ng, DeepLearning.AI and Stanford Online, [Machine Learning Specialization](https://www.deeplearning.ai/specializations/machine-learning/)
- Coursera, [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning)
