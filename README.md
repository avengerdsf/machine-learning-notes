# 机器学习学习笔记

本仓库按照吴恩达机器学习课程的学习顺序，整理监督学习与深度学习的核心概念、数学公式、结构示意图及代码示例。

## 目录

- [第一章：监督学习](#第一章监督学习)
- [第二章：深度学习](#第二章深度学习)
- [第三章：模型评估](#第三章模型评估)
- [第四章：决策树](#第四章决策树)
- [第五章：无监督学习](#第五章无监督学习)

## 第一章：监督学习

1. [线性回归模型](chapter_01_supervised_learning/01_learning_regression.md)：线性回归、代价函数、梯度下降与特征缩放。
2. [逻辑回归模型](chapter_01_supervised_learning/02_learning_logistic_regression.md)：Sigmoid 函数、决策边界、逻辑回归损失与梯度下降。
3. [过拟合问题](chapter_01_supervised_learning/03_learning_overfitting.md)：过拟合、欠拟合、正则化以及正则化后的线性回归和逻辑回归。

## 第二章：深度学习

1. [神经网络层](chapter_02_deep_learning/01_neural_network_layer.md)：单层与多层神经网络、前向传播以及 NumPy、TensorFlow 和 PyTorch 中的数据表示。
2. [构建神经网络](chapter_02_deep_learning/02_building_neural_network.md)：使用 PyTorch 构建、训练和执行神经网络。
3. [激活函数的选择](chapter_02_deep_learning/03_activation_function_selection.md)：Sigmoid、ReLU 与线性激活函数的特点和适用场景。
4. [Softmax 回归与多分类](chapter_02_deep_learning/04_softmax_regression_multiclass.md)：Softmax、多分类交叉熵、数值稳定实现与 PyTorch 示例。
5. [多标签分类问题](chapter_02_deep_learning/05_multilabel_classification.md)：多标签输出、独立 Sigmoid、二元交叉熵与 PyTorch 示例。

## 第三章：模型评估

1. [模型评估](chapter_03_model_evaluation/01_model_evaluation.md)：训练集与测试集划分、回归误差、分类误差以及泛化能力评估。
2. [模型的选择](chapter_03_model_evaluation/02_model_selection.md)：验证集划分、模型与超参数选择、偏差和方差诊断以及学习曲线。
3. [机器学习的开发过程](chapter_03_model_evaluation/03_machine_learning_development_process.md)：迭代开发、错误分析、数据改进、迁移学习、项目生命周期、伦理与分类指标。

## 第四章：决策树

1. [决策树](chapter_04_decision_trees/01_decision_tree.md)：猫分类示例、递归学习、熵、信息增益、连续特征的阈值选择和完整建树算法。
2. [One-hot 编码](chapter_04_decision_trees/02_one_hot_encoding.md)：多类别离散特征的二元化表示及其在决策树中的使用。
3. [回归树](chapter_04_decision_trees/03_regression_tree.md)：叶节点均值、方差下降以及使用动物特征预测体重的课程示例。
4. [随机森林](chapter_04_decision_trees/04_random_forest.md)：多棵决策树、放回抽样、Bagging、随机特征子集和预测聚合。
5. [XGBoost](chapter_04_decision_trees/05_xgboost.md)：Boosting、梯度修正、正则化目标函数以及分类和回归代码示例。

## 第五章：无监督学习

1. [K-means 聚类](chapter_05_unsupervised_learning/01_kmeans.md)：聚类问题、簇分配、质心更新、优化目标、随机初始化、聚类数量选择及 NumPy 完整实现。
2. [异常检测算法](chapter_05_unsupervised_learning/02_anomaly_detection.md)：高斯分布、参数估计、概率阈值、评估指标、特征选择、多元高斯分布及 NumPy 完整实现。
3. [推荐系统](chapter_05_unsupervised_learning/03_recommender_systems.md)：评分矩阵、协同过滤、二元标签、均值归一化、相关物品、基于内容的推荐和 NumPy 完整实现。
4. [PCA 主成分分析](chapter_05_unsupervised_learning/04_pca.md)：降维动机、数据预处理、SVD 主成分、投影重建、选择主成分数量及 NumPy 完整实现。
