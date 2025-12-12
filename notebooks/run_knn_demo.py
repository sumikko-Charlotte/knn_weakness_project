# A3项目：kNN算法弱点分析与反例构造
# 请帮我写完整的Python代码，展示kNN在三种情况下的失败：

# 案例1：棋盘分布（checkerboard pattern）
# 生成数据：创建类似国际象棋棋盘的数据分布
# 问题：kNN基于局部邻居投票，在棋盘分布中会失败

# 案例2：维度灾难（curse of dimensionality）
# 展示随着特征维度增加，kNN准确率下降
# 可视化维度 vs 准确率的关系

# 案例3：不同密度聚类（varied density clusters）
# 生成两个密度不同的聚类
# 展示kNN会偏向密集区域

# 要求：
# 1. 每个案例有独立的小节
# 2. 使用matplotlib可视化
# 3. 包含准确率计算
# 4. 对比其他算法（如决策树）

# -------------------------------
# kNN算法弱点案例1：棋盘分布模式（Checkerboard Pattern）
# -------------------------------

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs  # 添加这个

# 生成棋盘分布数据
def generate_checkerboard(n_samples=2000, noise=0.1):
    np.random.seed(42)
    X = np.random.uniform(0, 4, (n_samples, 2))  # 取[0,4]范围内的点
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)  # 棋盘黑白格
    X += np.random.normal(scale=noise, size=X.shape) # 添加噪声
    return X, y

X_cb, y_cb = generate_checkerboard()

# 划分训练集和测试集
X_cb_train, X_cb_test, y_cb_train, y_cb_test = train_test_split(X_cb, y_cb, test_size=0.3, random_state=42)

# 训练kNN和决策树
knn_cb = KNeighborsClassifier(n_neighbors=5)
dt_cb = DecisionTreeClassifier(random_state=42)
knn_cb.fit(X_cb_train, y_cb_train)
dt_cb.fit(X_cb_train, y_cb_train)

# 预测与准确率
y_cb_pred_knn = knn_cb.predict(X_cb_test)
y_cb_pred_dt = dt_cb.predict(X_cb_test)
acc_cb_knn = accuracy_score(y_cb_test, y_cb_pred_knn)
acc_cb_dt = accuracy_score(y_cb_test, y_cb_pred_dt)

print(f"案例1-棋盘分布 kNN准确率: {acc_cb_knn:.2f}，决策树准确率: {acc_cb_dt:.2f}")

# 可视化分类边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min() - 0.2, X[:,0].max() + 0.2
    y_min, y_max = X[:,1].min() - 0.2, X[:,1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=18)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_decision_boundary(knn_cb, X_cb_test, y_cb_test, f'kNN 分类结果 (acc={acc_cb_knn:.2f})')

plt.subplot(1,2,2)
plot_decision_boundary(dt_cb, X_cb_test, y_cb_test, f'决策树分类结果 (acc={acc_cb_dt:.2f})')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/decision_boundary_case1.png")
plt.close()

# 说明：
print("""
【案例1说明】
kNN在棋盘分布问题上表现较差，因为其只关注局部邻居，容易受相邻不同类别噪声的影响，导致分类边界错误。决策树能根据全局特征划分，效果更好。
""")

# -------------------------------
# kNN算法弱点案例2：维度灾难（Curse of Dimensionality）
# -------------------------------

from sklearn.datasets import make_classification

dims = [2, 5, 10, 20, 40, 60, 80, 100]
acc_knn_dim = []
acc_dt_dim = []

for d in dims:
    X_dim, y_dim = make_classification(n_samples=1200, n_features=d, n_informative=d, n_redundant=0, n_clusters_per_class=1, random_state=42)
    X_dtr, X_dte, y_dtr, y_dte = train_test_split(X_dim, y_dim, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(random_state=42)
    knn.fit(X_dtr, y_dtr)
    dt.fit(X_dtr, y_dtr)
    acc_knn_dim.append(accuracy_score(y_dte, knn.predict(X_dte)))
    acc_dt_dim.append(accuracy_score(y_dte, dt.predict(X_dte)))

plt.figure(figsize=(7,5))
plt.plot(dims, acc_knn_dim, marker='o', label='kNN')
plt.plot(dims, acc_dt_dim, marker='s', label='决策树')
plt.xlabel("特征维度数")
plt.ylabel("准确率")
plt.title("随着维度增加，模型准确率变化")
plt.legend()
plt.grid(True)
os.makedirs("results", exist_ok=True)
plt.savefig("results/dim_curse.png")
plt.close()

print("""
【案例2说明】
kNN随着特征维度增加，欧氏距离的判别能力减弱，分类准确率下降（维度灾难效应）。而决策树相对鲁棒，对高维数据影响较小。
""")

# -------------------------------
# kNN算法弱点案例3：不同密度的聚类（Varied Density Clusters）
# -------------------------------

from sklearn.datasets import make_blobs

# 生成不同密度的两个聚类
centers = [[-2,0], [2,0]]
cluster_std = [0.3, 1.2]  # 一类密集，一类稀疏
X_den, y_den = make_blobs(n_samples=[400,400], centers=centers, cluster_std=cluster_std, random_state=42)
X_den_tr, X_den_te, y_den_tr, y_den_te = train_test_split(X_den, y_den, test_size=0.3, random_state=42)

knn_den = KNeighborsClassifier(n_neighbors=5)
dt_den = DecisionTreeClassifier(random_state=42)
knn_den.fit(X_den_tr, y_den_tr)
dt_den.fit(X_den_tr, y_den_tr)
acc_knn_den = accuracy_score(y_den_te, knn_den.predict(X_den_te))
acc_dt_den = accuracy_score(y_den_te, dt_den.predict(X_den_te))
print(f"案例3-密度差异聚类 kNN准确率: {acc_knn_den:.2f}，决策树准确率: {acc_dt_den:.2f}")

# 可视化
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_decision_boundary(knn_den, X_den_te, y_den_te, f'kNN分类 (acc={acc_knn_den:.2f})')
plt.subplot(1,2,2)
plot_decision_boundary(dt_den, X_den_te, y_den_te, f'决策树分类 (acc={acc_dt_den:.2f})')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/varied_density.png")
plt.close()

print("""
【案例3说明】
kNN对密集类样本点判断更可靠，而在稀疏类别边界上容易受密集类别的邻居"多数投票"误导，导致偏向于密集簇。决策树则能够较好地区分不同类别。
""")

