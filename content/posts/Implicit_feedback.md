---
title: "Implicit feedback 下での推薦アルゴリズムに関わる論文"
date: 2021-03-24
draft: false
math: true
tags: ["machine learning", "recommendation", "paper"]
---

最近は推薦の問題に取り組むことが多く、特に実務の現場では implicit feedback という状況を扱う必要があるため関連する論文を読んで勉強していました。重要そうな論文をいくつかリストアップしたいと思います。

---
1. [Item Recommendation from Implicit Feedback](http://arxiv.org/abs/2101.08769)

    最近 arXiv に投稿されたレビュー論文です。Bayesian Personalized Ranking の考案者であるこの業界の第一人者、Steffen Rendle氏による解説です。Implicit feedback 下におけるアイテム推薦についてコンパクトにまとまっています。ウェブサイトで使う推薦モデルの構築では、ユーザーの行動ログからデータセットを作ることが多いのですが、その場合負例が集まりづらいという問題があります。ユーザーは興味のない対象にはそもそも関わらずログも残らないからです。このような状況下でモデルを訓練する際に行われる負例サンプリングについての解説や、サンプリングを回避するための手法である Gramian Trick についての解説が充実しています。

    以下に挙げる論文の多くはこのレビューの中で言及されています。

    またこの論文を読む際に、以下の記事を参考にさせて頂きました。

    ["Item Recommendation from Implicit Feedback"の紹介 | AI tech studio](https://cyberagent.ai/blog/research/publication_review/14483/)

2. [Folding: Why Good Models Sometimes Make Spurious Recommendations](https://dl.acm.org/doi/10.1145/3109859.3109911)

    推薦の「悪さ」をどうやって測るかについての研究です。行列分解などの embedding を利用するモデルをユーザーのフィードバックだけで学習すると、本来埋め込み空間内で遠くにあるべきアイテムを近くに配置してしまうと指摘しています。実際に toy dataset でそのような現象を可視化しており、負例サンプリングの必要性がわかる論文になっています。


3. [Collaborative Filtering for Implicit Feedback Datasets](http://ieeexplore.ieee.org/document/4781121/)

    行列分解を implicit feedback に拡張した論文です。ユーザーが反応しなかったアイテムを負例として加えること、それらに適当な重みを設けること、負例サンプリングを回避するために Gram 行列を上手く利用すること（Gramian trick）を提案しています。Alternating least squares という手法で最適化をしているため、iALS と呼ばれたりしています。

4. [SLIM: Sparse Linear Methods for Top-N Recommender Systems](http://glaros.dtc.umn.edu/gkhome/node/774)

    行列分解の特殊な場合として、$A=U^TW$ でユーザーの embedding 行列 $U$ を $A$ そのものに設定することを提案している論文です。パラメータは $W$ のみになるのでモデルとしては線形になります。$W$ がスパースになるよう $\ell_1$ 正則化を行います。一般にユーザーの行動履歴 $A$ もスパースなので、これで高速に推論を行うことができます。なお、実際には $\ell_2$ 正則化も課して $\ell_1$ とのバランスをとるので、いわゆる Elastic net になっています。

    SLIMは [RecSys 2019 のベストペーパー](http://arxiv.org/abs/1907.06902)で neural collaborative filtering を打ち破ることのできるモデルとしても紹介されていました。

    Elastic net の最適化手法として例えば引用されている以下の論文が勉強になります。$\ell_1$ 正則化がスパース性をもたらすことがよくわかります。

    [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929880/)


5. [A Generic Coordinate Descent Framework for Learning from Implicit Feedback](http://arxiv.org/abs/1611.04666)

    iALSで提案された Gramian Trick を Factorization Machine などに拡張しています。あまりちゃんと読んでません。扱っている問題は基本的には双線形のようなので、Coordinate descent で学習しています。

6. [Efficient Training on Very Large Corpora via Gramian Estimation](http://arxiv.org/abs/1807.07187)

    iALS で提案された Gramian trick を、一般のニューラルネットワークモデルに拡張した論文です。ユーザーの embedding ベクトル $u$ とアイテムの embedding ベクトル $v$ の内積、 $u^Tv$ によってターゲットを近似するようなモデル（dot-product モデルと呼ばれています）であれば使えます。この手のモデルは two-tower モデルとも呼ばれ、最近は [YouTube](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf) や [Twitter](https://www.semanticscholar.org/paper/Lessons-Learned-Addressing-Dataset-Bias-in-at-Virani-Baxter/54a5595575455a03da92fb0fe5a6513a8a4a25f4) などの有名企業の推薦モデルとしてよく使われている印象です。最近出た [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) でも利用が想定されています。[^1]
     一般の非線形なモデルでは iALS で提案された最適化手法が使えないのですが、そこを工夫してSGDで学習できるようにしたのがこの論文の貢献です。



---

最初のレビューで挙げられている BPR 系の pair-wise objective モデルだったり、softmax で直接分布の密度推定をするという手法ももう少し詳しく知っておきたいです。

[^1]: Two-tower モデルはニューラルネットワークの高い表現能力を生かしつつ、オンラインで高速に推論を行えるという利点を持っています。ターゲットを二つのベクトルの内積で近似するため、推論時には近似近傍探索を利用することができるからです。