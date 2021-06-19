---
title: "CatBoost でランキング学習に使われる Stochastic Rank について"
date: 2021-06-13T14:23:38+09:00
draft: true
math: true
tags: ["learning-to-rank"]
---

# 概要
CatBoost ではランキング学習用の objective として StochasticRank という関数が実装されている。
これによってnDCGなどの定番のランキングメトリックを最適化することができる。逆に、XGBoostにある LambdaMart などは実装されていないようである。

このStochasticRankについて調べようと思いドキュメントに載っていた論文を読んだのだが、書き方がなかなか数学的で読みづらかったので、自分なりの理解をまとめておこうと思う。

## 資料

- ドキュメント: [Ranking: objectives and metrics](https://catboost.ai/docs/concepts/loss-functions-ranking.html)
- 論文 [Ustimenko & Prokhorenkova 2020]: [StochasticRank: Global Optimization of Scale-Free Discrete Functions](https://arxiv.org/abs/2003.02122v2)
- [著者の発表動画](https://slideslive.com/38927767/stochasticrank-global-optimization-of-scalefree-discrete-functions?ref=speaker-18744-latest)

---

# ランキング学習
ランキング学習はアイテム単独のスコアではなく複数アイテムの「良い並び順」を学習・評価する手法である。
例えば検索や推薦の分野で、ユーザーに提示するアイテムの候補を抽出した後、個々のユーザーに対してより適した順番でそれらを並び替えるという状況で使われる。
ユーザーはこのアイテムリストの上から目にするため、リストの先頭にユーザーにとって良いアイテムを持ってくることが重要となる。

今、アイテムが $n$ 個あり、モデルが各アイテムに対して $z=(z_1,z_2,\dots,z_n)$ というスコアを出力するとしよう。
そして $n$ 個のアイテムをこのスコアの大きい順に並び替える。先頭を $i=1$ として、上から $k$ 個を最終的なモデルの予測結果のリストとする。
このリストの良さを図るメトリックとして、例えば次の discounted cumulative gain (DCG) という関数が使われる [^1]

`
$$
\mathrm{DCG@k} = \sum_{i=1}^{\mathrm{min}\{k, n\}}\frac{2^{r_i}-1}{2^4\log_2(1+i)}\,.
$$
`
ここで $r_i$ はそのアイテムの真の relevance (ground truth) である。これはクリックで学習していれば $0$ or $1$ だし、商品レビューの評点のようなものなら $r=1,2,...5$ などとなる。大きい方がより relevant であれば良い。単なる relevance の和をとるのではなく、各順位のログ $\log_2(1+i)$ で割ることによってリストの先頭をより大きい重みで評価するようになっている。[^3]
またこれを normalize した nDCG (normalized DCG) というメトリックもよく使われる
`
$$
\mathrm{nDCG@k} = \frac{\mathrm{DCG@k}}{\mathrm{Ideal\, DCG@k}}
$$
`
ここで分母の $\text{Ideal }\mathrm{DCG@k}$ は答えを知っていると仮定した時の理想的な DCG のことで、 ground truth $r_i$ の通りに並び替えることで得られる。

メトリックはこれ以外にも色々あるが本質ではないのでこれ以上は割愛。

---
# ランキングメトリックの最適化
単純な回帰や分類の目的関数 (例えば Mean squared error や logistic loss) に比べて、ランキング学習のメトリックを目的関数とした最適化は難しい。
スコア $z$ を変化させて行った時にランキングメトリックが変化するのは、ある二つのアイテム $i$ と $j$ で $z_i$ と $z_j$ の大小関係が変わる場合だけである。
そのため、ランキングメトリックは model prediction $z$ に対して piecewise-constant、つまり、大体の領域でフラットで、時々不連続に変化するような関数になっている。
このために、単純に目的関数の微分を取って gradient descent するという手段が使えない。

以下ではこの問題にアプローチする代表的な方法を二つ紹介する。

## 代理の目的関数を最適化する
例えば LambdaRank・LambdaMart のような手法がこれである。この手法は目的関数を直接定義せず、その微分だけを定義して計算する。
これで本当に最適化したいランキングメトリックを最適化できるかはわからないが、 empirical にはとてもうまくいくようである。
詳細は例えば以下の文献を参照。

[From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf)



## 目的関数を smoothing する
ランキングメトリックが不連続に変化する部分を smooth になるよう変形する手法。
代表的なものとして SoftRank がある。

[SoftRank: optimizing non-smooth rank metrics](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SoftRankWsdm08Submitted.pdf)

このような手法では $z$ に微小なノイズを加えて smoothing する。式で書くと以下のような感じ。
`
$$
  L^{soft}(z, r) = \lim_{\sigma \to +0}\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma)}L(z + \sigma\epsilon, r)\,,
$$
`
抽象的に書くとわかりづらいので、$n=2$ の最もシンプルなケースで考えてみる。

`
\begin{align}
L^{soft}
&=
  \mathrm{MRR}(z_1 > z_2)
  \int_{-\infty}^{\infty} d\epsilon_1 \int_{-\infty}^{\epsilon_1 + (z_1-z_2)/\sigma}d\epsilon_2
  \mathcal{N}(\epsilon_1|0, 1)\mathcal{N}(\epsilon_2|0, 1) \\
&+
  \mathrm{MRR}(z_1 < z_2)\int_{-\infty}^{\infty} d\epsilon_1 \int_{\epsilon_1 + (z_1-z_2)/\sigma}^\infty d\epsilon_2
  \mathcal{N}(\epsilon_1|0, 1)\mathcal{N}(\epsilon_2|0, 1)\,.
\end{align}
`

$\sigma\to0$ で $z_1$ と $z_2$ の大小関係に応じて積分の片方が落ちるので、元々のMRRに収束する。この微分は簡単に計算できて

`
$$
\frac{\partial L^{soft}}{\partial z_1}=-\frac{\partial L^{soft}}{\partial z_2}
=\left(\mathrm{MRR}(z_1 > z_2) - \mathrm{MRR}(z_1 < z_2)\right)\frac{1}{2\sqrt{\pi}\sigma}\exp\left(-\frac{(z_1-z_2)^2}{4\sigma^2}\right)\,.
$$
`
$\sigma\to0$ でデルタ関数的な振る舞いをすることがわかる。
このように、ガウシアンノイズによる smoothing はメトリックの差×(exponential smoothing)という形でランキングメトリックの微分を近似している。

---
# StochasticRank

## Tie ambiguity

本題である StochasticRank は上のようなノイズを使った smoothing 手法の拡張である。
上では誤魔化していたが、nDCG の定義には一点曖昧な部分がある。それは $z_i=z_j$ となるようなタイの取り扱いである。

ランキングメトリックの計算ではスコア $z$ に応じてアイテムを並び替えるわけだが、ではモデルの予測が同じ $z_i=z_j$ となるようなアイテムはどう並べたら良いだろうか？
例えば [scikit-learn の nDCG のドキュメント](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)には、
> by default ties are averaged, so here we get the average (normalized)

と書いてあり、可能な並び替えの平均を最終的な値として出力するようにしている。


一方 [CatBoost](https://catboost.ai/docs/references/ndcg.html#calculation) では
> Top samples are either the samples with the largest approx values or the ones with the lowest target values if approx values are the same

とあり、タイの場合は relevance の低い順 = 最も nDCG が悪くなる順 (*worst permutation*) で並べるようである。

モデルの予測スコア $z$ が連続値ならナイーブには $z_i=z_j$ となるような稀なケースは気にしなくて良いのではないかと思うかもしれない。しかし、tree 系のアルゴリズムでは同じ leaf に属するアイテムは同じスコアになるので、このようなことが発生しやすい。また、ノイズを付与して smoothing をして学習を行う場合、微分に対する本質的な寄与は $z_i=z_j$ 付近の関数の形からくる。そのためタイの取り扱いはそのまま性能に影響を与える。

では、上で紹介した $L^{soft}$ はタイをどのように扱っているだろうか？上に書いた $n=2$ の例で $z_1=z_2$ とすると
`
$$
L^{soft} = 0.5\times\mathrm{MRR}(z_1 > z_2) + 0.5\times\mathrm{MRR}(z_1 < z_2)
$$
`
となり、平均を取る手法を採用していることがわかる。
となると、これを使って worst permutation で定義された nDCG を最適化することはできない。
平均を取るのが良いのか、worst permutation を取るのが良いのかを簡単に決めることはできないと思うが、[Ustimenko & Prokhorenkova 2020]では、worst permutation の方がタイを解消する方向に学習をブーストしやすいので良いのではないかと述べられている。

## StochasticRank objective

StochasticRank は worst permutation でもノイズを使った smoothing が行えるように SoftRank 系の手法を拡張したものである。式で書くと

`
$$ L^{stochastic}(z, r)\equiv\lim_{\mu\to\infty}\lim_{\sigma\to 0}
\mathbb{E}_{\epsilon \sim \mathcal{N}(-\mu r, \mathbb{1})}L(z + \sigma\epsilon, r)\,.
$$
`
ここで $r=(r_1, r_2,...)$ はリストの $i$ 番目のアイテムの relevance である。SoftRank との違いは、ガウシアンノイズの平均を $0$ ではなく、$-\mu r$ に取っているところ。これはモデルスコア $z$ を最も悪化させる方向になっている。そのため、このノイズで worst permutation に対応する smoothing になる。
感覚を掴むため、もう一度 $n=2$ の例で書いてみると
`
\begin{align}
L^{stochastic}
  &=
  \mathrm{MRR}(z_1 > z_2)\int_{-\infty}^{\infty} d\epsilon_1 \int_{-\infty}^{\epsilon_1 + (z_1-z_2)/\sigma -\mu(r_1-r_2)}d\epsilon_2
  \mathcal{N}(\epsilon_1|0, 1)\mathcal{N}(\epsilon_2|0, 1) \notag \\
  &+
  \mathrm{MRR}(z_1 < z_2)\int_{-\infty}^{\infty} d\epsilon_1 \int_{\epsilon_1 + (z_1-z_2)/\sigma -\mu(r_1-r_2)}^\infty d\epsilon_2
  \mathcal{N}(\epsilon_1|0, 1)\mathcal{N}(\epsilon_2|0, 1)\,.
\end{align}
`
$z_1=z_2$ の場合は $\mu\to\infty$ で $r_1$ と $r_2$ の大小関係に応じて積分の一方が残る。$r_1 < r_2$ の場合は $L^{stochastic}(z_1=z_2) = \mathrm{MRR}(z_1 > z_2)$ となり、確かに worst case になっている。

微分は
`
\begin{align}
  \frac{\partial L^{stochastic}}{\partial z_1}
  &=
  -\frac{\partial L^{stochastic}}{\partial z_2} \notag \\
  &=
  \left(\mathrm{MRR}(z_1 > z_2) - \mathrm{MRR}(z_1 < z_2)\right)
  \frac{1}{2\sqrt{\pi}\sigma}\exp\left(-\frac{1}{4}
  \left(\frac{z_1-z_2}{\sigma} - \mu(r_1 - r_2)\right)^2
  \right)\,.
\end{align}
`
SoftRank の式と比べると、exponential smoothing の中心が worst 方向にずれていることがわかる。特に、$z_1=z_2$ 付近では、SoftRank に比べて additional な supression factor $\exp(-\mu^2/4) $ がついている。

## 実験結果


[^1]: 分母の$2^4$は [Ustimenko & Prokhorenkova 2020]に倣ってつけたが特に深い意味はないように思われる。
[^3]: relevance の冪乗を使っている部分に深い意味があるわけではなく、$2^{r_i}-1$ を $r_{i}$ に置き換えても良い。問題に応じて適切なものを用いる。