---
title: "カーネルSVMをシンプルに実装する"
emoji: "📝"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "SVM", "機械学習", "数理最適化"]
published: false
---

## TL;DR
- 線形SVMと同様に、カーネルSVMを実装している記事もあまり見かけなかったのでpythonで実装してみました。
- カーネルSVMの主問題を、L2正則化経験リスク最小化問題として勾配降下法で解きます。
- 計算速度はscikit-learnのSVCのような工夫されたアルゴリズムには及びません。

## 本記事が実装するSVM
前回の線形SVMの内容を前提として進めたいと思います。
2クラス分類を行う、カーネルSVMの実装を行います。
実装はこちらにあります。
@[card](https://github.com/obizip/SVM)

## 前回の記事の振り返り
:::message
簡単のために特徴量$\bm{x}$と重み$\bm{w}$を次のように拡張します。
$$\tilde{\bm{x}} = [\bm{x}, 1]^\top, \tilde{\bm{w}} = [\bm{w}, b]$$
**以下では単に$\tilde{\bm{x}}$を$\bm{x}$、$\tilde{\bm{w}}$を$\bm{w}$と表記します。**
また、特徴量を$\bm{x}_1, \cdots, \bm{x}_n \in \mathbb{R}^d$、クラスラベルを$y_1, \cdots, y_n \in \{-1, 1\}$とします。
:::

### 線形SVM
ここで、線形分離不可能なデータセットに対して、$\xi$だけ分類境界からの誤差を許容するようにハードマージンSVMを一般化した手法を[ソフトマージンSVM](https://ja.wikipedia.org/wiki/%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88%E3%83%99%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%9E%E3%82%B7%E3%83%B3#%E3%82%BD%E3%83%95%E3%83%88%E3%83%9E%E3%83%BC%E3%82%B8%E3%83%B3)と呼びます。
基本的なソフトマージンSVMは$\bm{y}_i \langle \bm{w}, \bm{x_i} \rangle = 0$で表されるような一次関数でデータセットを分離するため**線形SVM**とも呼ばれます。

### 主問題

定数$\lambda \in \mathbb{R}$として、線形SVMの主問題は次のように表せます。

$$\min_w{\frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\xi_i} \quad \text{s.t.} \quad \xi_i \ge 0, \quad \bm{y}_i \langle \bm{w}, \bm{x_i} \rangle \ge 1 - \xi_i$$

### 経験リスク最小化
線形SVMの主問題は経験リスク$R$を用いて次のように、ヒンジ損失によるL2正則化経験リスク最小化問題として表すことができます。

$$\begin{gather*}\min_w{R(\bm{w})}\\ \text{where} \quad R(\bm{w}) := \frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\max\{0, 1-y_i \langle \bm{w}, \bm{x_i} \rangle \}\end{gather*}$$

### 線形SVMの限界
この線形SVMで、次のようなmoonデータセットを予測しようとすると、データセットの分布に沿った境界にはなっていません。
このように、上で扱った一次関数で分離するSVM(線形SVM)では限界があります。
![dataset_moon](/images/2024-07-04-linear_svm/dataset_moon.png)

## カーネルSVMとは
特徴量$\bm{x}$を特徴写像$\Phi$によって高次元な特徴空間に写し、この高次元空間上で一次関数により分類を行う方法を**カーネルSVM**と呼びます。

直感的には、データの特徴量を高次元に写すことで、表現力が増し、より分類しやすくなりそうです。しかし、特徴空間の次元が大きければ大きいほど、その空間での特徴量$\Phi(\bm{x})$の次元も大きくなり、計算にコストがかかってしまいます。

ここで、特殊な関数である**カーネル関数**$\kappa$と呼ばれる2変数関数を用いることで、特徴空間中の特徴量の**明示的な計算を経由せず**に、特徴空間における**内積をデータから直接計算することができます**(このことを**カーネルトリック**といいます)。
今回は、次のようにカーネル関数が表されるとします。
$$\kappa(\bm{u}, \bm{v}) = \left\langle\Phi(\bm{u}), \Phi(\bm{v})\right\rangle$$

## カーネルSVMの経験リスク関数
線形SVMの経験リスク$R$は次のようでした。
$$R(\bm{w}) := \frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\max\{0, 1-y_i \langle \bm{w}, \bm{x_i} \rangle \}$$
今、特徴量$\bm{x}_i$を、特徴写像$\Phi$によって高次元な特徴空間に写した$\Phi(\bm{x}_i)$として計算することを考えます。すると、上式は次のようになります。
$$R(\bm{w}) := \frac{\lambda}{2}||\bm{w}||^2 + \frac{1}{n}\sum^n_{i=1}\max\{0, 1-y_i \langle \bm{w}, \Phi(\bm{x_i}) \rangle\}$$
このまま、前回の線形SVMと同様に勾配降下法を用いて解くことができればよいのですが、上で述べた通り、高次元空間での特徴量$\Phi(\bm{x}_i)$の計算には、大きなコストがかかります。

ここで、一般的には双対問題を考えますが、今回は別の手法をとります。
この経験リスク$R$の重み$\bm{w}$は、**リプレゼンター定理**を用いて次のように表すことができます。

$\bm{\alpha} = [\alpha_1, ..., \alpha_n] \in \mathbb{R}^n$を用いて$\bm{w}$を次のように表すことができる。
$$\bm{w} = \sum^n_{j=1} \alpha_j\Phi(\bm{x}_j)$$
これを踏まえると経験リスク$R$は次のようになる.

ただし、$K = [\bm{k}_1, ..., \bm{k}_n] = [\kappa(\bm{x}_i, \bm{x}_j)]_{i, j},~\quad i, j \in [n]$とする。

$$\begin{align*}
	R(\bm{\alpha}) & = \frac{\lambda}{2} \left\|\sum^n_{j=1} \alpha_j\Phi(\bm{x}_j)\right\|^2 + \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1- y_i\left\langle\sum^n_{j=1} \alpha_j\Phi(\bm{x}_j), \Phi(\bm{x}_i)\right\rangle\right\} \\
	           & = \frac{\lambda}{2}\left\|\sum^n_{j=1} \alpha_j\Phi(\bm{x}_j)\right\|^2
	+ \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1-y_i\sum^n_{j=1} \alpha_j\left\langle\Phi(\bm{x}_j), \Phi(\bm{x}_i)\right\rangle \right\} \\
	           & = \frac{\lambda}{2}\left\langle\sum^n_{i=1} \Phi(\bm{x}_i)\alpha_i, \sum^n_{j=1} \Phi(\bm{x}_j)\alpha_j\right\rangle
	+ \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1-y_i \alpha_j \sum^n_{j=1}\left\langle\Phi(\bm{x}_j), \Phi(\bm{x}_i)\right\rangle\right\} \\
	           & = \frac{\lambda}{2}\sum^n_{i=1}\sum^n_{j=1}\alpha_i\left\langle \Phi(\bm{x}_i),  \Phi(\bm{x}_j)\right\rangle\alpha_j
	+ \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1-y_i \alpha_j \sum^n_{j=1}\left\langle\Phi(\bm{x}_j), \Phi(\bm{x}_i)\right\rangle\right\} \\
	           & = \frac{\lambda}{2}\sum^n_{i=1}\sum^n_{j=1}\alpha_i\kappa(\bm{x}_i, \bm{x}_j)\alpha_j
	+ \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1-y_i \alpha_j \sum^n_{j=1}\kappa(\bm{x}_j, \bm{x}_i)\right\}          \\
	           & = \frac{\lambda}{2}\left\langle\bm{\alpha}, K \bm{\alpha}\right\rangle
	+ \frac{1}{n}\sum^n_{i=1} \max\left\{0, 1-y_i\left\langle\bm{\alpha}, \bm{k}_i\right\rangle\right\}
\end{align*}$$

よって経験リスク$R$の勾配を計算すると次のようになる。

$$\begin{align*}
    \nabla R(\bm{\alpha}) & = \lambda K \bm{\alpha}
    + \frac{1}{n}\sum^n_{i=1} \begin{cases} -y_i\bm{k}_i &\text{if } 1-y_i \langle \bm{\alpha}, \bm{k}_i \rangle \ge 0 \\ 0 &\text{otherwise}\end{cases} \\
\end{align*}$$

### 経験リスク最小化と最急降下法
線形SVMと同様に、経験リスク$R$が[凸関数](https://ja.wikipedia.org/wiki/%E5%87%B8%E9%96%A2%E6%95%B0)であることから、単純に[最急降下法](https://ja.wikipedia.org/wiki/%E6%9C%80%E6%80%A5%E9%99%8D%E4%B8%8B%E6%B3%95)を用いることで最適解を得ることができます。
つまり、$t$ステップ目の重み$\bm{\alpha}^{(t)}$は、学習率$\eta \in \mathbb{R}$を定めて、前ステップの重み$\bm{\alpha}^{(t-1)}$と、経験リスク$R$の勾配から、次のように決めていけば良いことになります。
$$\bm{\alpha}^{(t)} = \bm{\alpha}^{(t-1)} - \eta \nabla R(\bm{\alpha}^{(t-1)})$$
このステップを十分に繰り返すことで、重み$\bm{\alpha}$は経験リスク$R$を最小とする最適解に近づいていきます。

### カーネルSVMでの予測
テストデータ$\tilde{\bm{x}}_1, \cdots, \tilde{\bm{x}}_m \in \mathbb{R}^d$に対して、勾配降下法で得た$\alpha$を用いて予測値を計算するには次のようにします。 ただし、$\tilde{K} = [\tilde{\bm{k}}_1, ..., \tilde{\bm{k}}_m] = [\kappa(\bm{x}_i, \tilde{\bm{x}}_j)]_{i, j}, \quad (i, j) \in [n]\times[m]$とします。

$$\begin{align*}
	i \in [m], \quad \hat{\bm{y}} & = [\left\langle\bm{w}, \Phi(\tilde{\bm{x}}_i)\right\rangle]_i                                                            \\
	                          & = \left[\left\langle\sum^n_{j=1}\alpha_j \Phi(\bm{x}_j) , \Phi(\tilde{\bm{x}}_i)\right\rangle\right]_i \\
	                          & = \left[\sum^n_{j=1} \alpha_j\left\langle\Phi(\bm{x}_j), \Phi(\tilde{\bm{x}}_i)\right\rangle \right]_i \\
	                          & = \left[\left\langle\bm{\alpha}, \tilde{\bm{k}}_i\right\rangle\right]_i                               \\
\end{align*}$$
