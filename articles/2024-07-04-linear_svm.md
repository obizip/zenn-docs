---
title: "線形SVMをシンプルに実装する"
emoji: "📝"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "SVM", "機械学習", "数理最適化"]
published: true
---

## TL;DR
- SVMを実装している記事はあまり見かけなかったのでpythonで実装してみました。
- SVMの主問題を、L2正則化経験リスク最小化問題として勾配降下法で解きます。
- 計算速度はscikit-learnのSVCのような工夫されたアルゴリズムには及びません。

## 本記事が実装するSVM
2クラス分類を行う、基本的なソフトマージンSVMの実装を行います。
**基本的なソフトマージンSVMは一次関数で分類するため、線形SVMとも呼ばれます**。

実装はこちらにあります。
@[card](https://github.com/obizip/SVM)
カーネル関数を取り入れたソフトマージンSVMであるカーネルSVMは次回実装します。

## SVMのざっくりとした理論
<!-- $n$個の特徴量$\bm{x}_i \in \mathbb{R}^d$とクラスラベル$y_i \in \{-1, 1\}$ -->
特徴量$\bm{x} \in \mathbb{R}^d$と重み$\bm{w} \in \mathbb{R}^d$とバイアス$b \in \mathbb{R}$に対して、次のような一次関数を考えます。
$$f(\bm{x}) = \langle \bm{w}, \bm{x} \rangle + b$$

この一次関数に対して、クラスが$-1, +1$のどちらかであるとし、$f(\bm{x}) > 0$であるとき、$\bm{x}$のクラスを$+1$と予測し、$f(\bm{x}) < 0$であるとき、$\bm{x}$のクラスを$-1$と予測するとします。


ここで、より予測精度の高い一次関数を引くには、重み$\bm{w}$とバイアス$b$をどのように決めれば良いでしょうか。

一般にクラスを分類する境界$f(\bm{x}) = 0$を**分類境界**と呼び、
分類境界を挟んで2つのクラスがどれくらい離れているかを**マージン**と呼びます。

[SVM(サポートベクトルマシン)](https://ja.wikipedia.org/wiki/%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E3%83%99%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%9E%E3%82%B7%E3%83%B3)はこのマージンを最大化することで、より予測精度の高い一次関数を求めます。

さて、今、簡単のために特徴量$\bm{x}$と重み$\bm{w}$を次のように拡張します。
$$\tilde{\bm{x}} = [\bm{x}, 1]^\top, \tilde{\bm{w}} = [\bm{w}, b]$$
すると、識別境界は
$$f(\bm{x}) = \langle \bm{w}, \bm{x} \rangle + b = \langle \tilde{\bm{w}}, \tilde{\bm{x}} \rangle$$
と簡潔に表すことができます。

:::message
**以下では単に$\tilde{\bm{x}}$を$\bm{x}$、$\tilde{\bm{w}}$を$\bm{w}$と表記します。**
また、特徴量を$\bm{x}_1, \cdots, \bm{x}_n \in \mathbb{R}^d$、クラスラベルを$y_1, \cdots, y_n \in \{-1, 1\}$とします。
:::

### ハードマージンSVMの主問題
データセットがある一次関数でクラスを完全に分離できるとき、それを[線形分離可能](https://ja.wikipedia.org/wiki/%E7%B7%9A%E5%BD%A2%E5%88%86%E9%9B%A2%E5%8F%AF%E8%83%BD)であるといいます。データセットを**線形分離可能であると仮定したときのSVM**を[ハードマージンSVM](https://ja.wikipedia.org/wiki/%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E3%83%99%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%9E%E3%82%B7%E3%83%B3#%E3%83%8F%E3%83%BC%E3%83%89%E3%83%9E%E3%83%BC%E3%82%B8%E3%83%B3)と呼びます。

ハードマージンSVMの主問題は次のように表せます。
$$\min_w{||\bm{w}||^2} \quad \text{s.t.} \quad y_i \langle \bm{w}, \bm{x}_i \rangle \ge 1$$

### ソフトマージンSVMの主問題
一般にデータセットが線形分離可能であることは少ないです。そのため、$\xi$だけ分類境界からの誤差を許容するようにハードマージンSVMを一般化した手法を[ソフトマージンSVM](https://ja.wikipedia.org/wiki/%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E3%83%99%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%9E%E3%82%B7%E3%83%B3#%E3%82%BD%E3%83%95%E3%83%88%E3%83%9E%E3%83%BC%E3%82%B8%E3%83%B3)と呼びます。

定数$\lambda \in \mathbb{R}$として、ソフトマージンSVMの主問題は次のように表せます。
$$\min_w{\frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\xi}_i \quad \text{s.t.} \quad \xi_i \ge 0, \quad \bm{y}_i \langle \bm{w}, \bm{x}_i\rangle \ge 1 - \xi_i$$

### ソフトマージンSVMと経験リスク最小化
**ここから一般的にSVMを解くために用いられる手法とは違うものになります。**

人工変数$\xi \in \mathbb{R}$に対して、任意の単調増加な関数$g: \mathbb{R} \rightarrow \mathbb{R}$で次の2つは等価になります。
$$\min_z{g(\max\{0, 1-z\})} \iff \min_{z, \xi}{g(\xi)} \quad \text{s.t.} \quad \xi \ge 0 \quad z \ge 1 - \xi$$

この同値関係から、ソフトマージンSVMの主問題は次のように書き換えることができます。
$$\begin{gather*}\min_w{R(\bm{w})}\\ \text{where} \quad R(\bm{w}) := \frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\max\{0, 1-y_i \langle \bm{w},  \bm{x}_i \rangle\}\end{gather*}$$


ここで、$\max\{0, 1-y_i \langle \bm{w},  \bm{x}_i \rangle\}$は[ヒンジ損失](https://en.wikipedia.org/wiki/Hinge_loss)と呼ばれる[損失関数](https://ja.wikipedia.org/wiki/%E6%90%8D%E5%A4%B1%E9%96%A2%E6%95%B0)です。
すると、この式は、深層学習でも用いられている正則化[経験リスク最小化(Empirical risk minimazation)](https://en.wikipedia.org/wiki/Empirical_risk_minimization)と呼ばれる手法と一致します。

### ソフトマージンSVMと最急降下法
ソフトマージンSVMの最小化問題を解く方法の一つとして、経験リスク$R$が[凸関数](https://ja.wikipedia.org/wiki/%E5%87%B8%E9%96%A2%E6%95%B0)であることから、単純に[最急降下法](https://ja.wikipedia.org/wiki/%E6%9C%80%E6%80%A5%E9%99%8D%E4%B8%8B%E6%B3%95)を用いることで最適解を得ることができます。
つまり、$t$ステップ目の重み$\bm{w}^{(t)}$は、学習率$\eta \in \mathbb{R}$を定めて、前ステップの重み$\bm{w}^{(t-1)}$と、経験リスク$R$の勾配から、次のように決めていけば良いことになります。
$$\bm{w}^{(t)} = \bm{w}^{(t-1)} - \eta \nabla R(\bm{w}^{(t-1)})$$

このステップを十分に繰り返すことで、重み$\bm{w}$は経験リスク$R$を最小とする最適解に近づいていきます。

ここで、経験リスク$R$の(劣)勾配は次のように計算することができます。
$$\nabla R(\bm{w}) = \lambda \bm{w} + \frac{1}{n} \sum^n_{i=1}\begin{cases} -y_i\bm{x}_i &\text{if } 1-y_i \langle \bm{w}, \bm{x}_i \rangle \ge 0 \\ 0 &\text{otherwise}\end{cases}$$

## ソフトマージンSVMの実装
### 単純な実装
さて、実装としては、**ソフトマージンSVMと最急降下法**の部分をプログラムに書き起こせば良いことになります。

まず、次のようなデータセットを用意しました。
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


np.random.seed(42)

N = 100
n_features = 2
X, y = make_blobs(random_state=8,
                  n_samples=N,
                  n_features=n_features,
                  cluster_std=3,
                  centers=2)

y = y * 2 - 1 # change y_i {0, 1} to {-1, 1}

plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.plot()
```
![](/images/2024-07-04-linear_svm/dataset1.png)

次に、ソフトマージンSVMを実装します。
```python
import numpy.linalg as LA


X = np.c_[X, np.ones(X.shape[0])] # set bias

# regularization parameter
lam = 0.01

# empirical risk
R = lambda w_t: 0.5 * lam * LA.norm(w_t, ord=2)**2 + np.where(1 - y * (X @ w_t) >= 0, 1 - y * (X @ w_t), 0).mean()

# gradient of empirical risk
dR = lambda w_t: lam * w_t + ((np.where(1 - y * (X @ w) >= 0, 1, 0) * -y).reshape(-1, 1) * X).mean(axis=0)

# learning rate
learning_rate = 0.01

# amount of iteration
n_iters = 5000

# weights
w = np.ones((X.shape[1]))

# gradient descent
for i in range(n_iters):
    w = w - learning_rate * dR(w)
    if (i + 1) % 1000 == 0:
        print(f"{i+1:4}: R(w) = {R(w)}")

print(f"weights: {w}")
# 1000: R(w) = 0.2580498763840658
# 2000: R(w) = 0.2518440922366277
# 3000: R(w) = 0.25032312599721734
# 4000: R(w) = 0.24936754026280172
# 5000: R(w) = 0.24909738755742183
# weights: [ 0.08554331 -0.4618898   1.69628161]
```

次のようにクラスを分離していることがわかります。
```python
plt.figure(figsize=(8, 7))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

# w0*x + w1*y + w2 = 0
# y = - (w0*x + w2) / w1
plt.plot(X[:, 0], - (w[0] * X[:, 0] + w[2]) / w[1])
```
![plot_boundary1](/images/2024-07-04-linear_svm/plot_boundary1.png)

### sklearn-APIに近い実装
単純な実装では訓練データとテストデータが分かれておらず、機械学習のモデルとしては不十分です。
より実用的である、scikit-learnのAPIに近い実装は次のようになります。
```python
import numpy as np
import numpy.linalg as LA


class LinearSVM:
    def __init__(
        self,
        lam: float = 0.01,
        n_iters: int = 5000,
        learning_rate: float = 0.01,
        bias=True,
        verbose=False,
    ):
        self.lam = lam
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.bias = bias
        self.verbose = verbose
        self._w = None

    def _empirical_risk(self, X: np.ndarray, y: np.ndarray, w_t: np.ndarray) -> np.ndarray:
        regularzation = 0.5 * self.lam * LA.norm(w_t, ord=2) ** 2
        loss = np.where(1 - y * (X @ w_t) >= 0, 1 - y * (X @ w_t), 0).mean()
        return regularzation + loss

    def _empirical_risk_grad(self, X: np.ndarray, y: np.ndarray, w_t: np.ndarray) -> np.ndarray:
        regularzation_grad = self.lam * w_t
        loss_grad = ((np.where(1 - y * (X @ w_t) >= 0, 1, 0) * -y).reshape(-1, 1) * X).mean(axis=0)
        return regularzation_grad + loss_grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.bias:
            X = np.c_[X, np.ones(X.shape[0])]

        # weights
        w = np.ones((X.shape[1]))

        # gradient descent
        for i in range(self.n_iters):
            w = w - self.learning_rate * self._empirical_risk_grad(X, y, w)
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"{i+1:4}: R(w) = {self._empirical_risk(X, y, w)}")
        self._w = w

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.bias:
            X = np.c_[X, np.ones(X.shape[0])]
        return X @ self._w

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores > 0, 1, -1)
```

実際に使ってみましょう。
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


np.random.seed(42)

N = 500
n_features = 2
X, y = make_blobs(random_state=8,
                  n_samples=N,
                  n_features=n_features,
                  cluster_std=3,
                  centers=2)

y = y * 2 - 1 # change y_i {0, 1} to {-1, 1}

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearSVM()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"ACC: {accuracy_score(y_test, preds)}")
# ACC: 0.936

plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
w = model._w
# w0*x + w1*y + w2 = 0
# y = - (w0*x + w2) / w1
plt.plot(X[:, 0], - (w[0] * X[:, 0] + w[2]) / w[1])
```
![plot_boundary2](/images/2024-07-04-linear_svm/plot_boundary2.png)

正解率が```ACC: 0.936```となり、うまく予想できています。

## 線形SVMの限界
例えば次のようなmoonデータセットで予測してみましょう。
```python
from sklearn.datasets import make_moons

N = 500
n_features = 2
X, y = make_moons(n_samples=N, noise=0.1, random_state=1)
y = y * 2 - 1 # change y_i {0, 1} to {-1, 1}
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearSVM()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"ACC: {accuracy_score(y_test, preds)}")
# ACC: 0.896

plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
w = model._w
plt.plot(X[:, 0], - (w[0] * X[:, 0] + w[2]) / w[1])
```
![dataset_moon](/images/2024-07-04-linear_svm/dataset_moon.png)
正解率は```ACC: 0.896```とそこまで悪くはありませんが、実際の分類境界を見ると、データセットの分布に沿った境界にはなっていません。

このように、上で扱った一次関数で分離するSVM(線形SVM)では限界があります。
この限界を解消するために、カーネル関数を用いたSVM(カーネルSVM)があります。この実装は次回の記事で紹介したいと思います。
