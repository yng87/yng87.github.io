---
title: "KaTeX"
date: 2021-03-20T10:41:06+09:00
draft: false
katex: true
---

Markdown に LaTeX 形式のテキストを埋め込んだときに発生する問題。


## アンダースコアがhtmlタグに変わる
---

```
$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}.$$
```
↓
```html
$$p(class = j) = \frac{\exp(a_{j})}{\sum<em>j \exp(a</em>{j})}.$$
```
KaTeX が上手くいかない時がある。

インライン \(j\)

$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}.$$


$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}$$


---

$$
p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}
$$
