---
title: "Hugo + Math equation"
date: 2021-03-20T10:41:06+09:00
draft: false
math: true
tags: ["hugo"]
---


## アンダースコアがhtmlタグに変わる
---
Markdown に LaTeX 形式のテキストを埋め込んだときに発生する問題。

MathJax や KaTexなどを使うと Hugo のコンテンツで数式を扱うことができる。しかし Hugo の markup は Markdown のアンダースコアで囲われた部分をイタリック体にするので、数式内に二つアンダースコア`_` があると、`<em>...</em>` が挿入されてしまう。

```
$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}.$$
```
↓
```html
$$p(class = j) = \frac{\exp(a_{j})}{\sum<em>j \exp(a</em>{j})}.$$
```

いろいろな workaround がありそうだが、以下のページを真似するのが簡単だった。数式のレンダリングには MathJax 3 を使う。

[Render LaTeX math expressions in Hugo with MathJax 3 · Geoff Ruddock](https://hatenablog-parts.com/embed?url=https://geoffruddock.com/math-typesetting-in-hugo/)

必要なページでのみ MathJax が動くように、`header.html` では以下のように MathJax を挿入した
```html
	{{ if and (isset .Params "math") (eq .Params.math true)}}
	{{ partial "mathjax_support.html" . }}
	{{ end }}
```

これで `math: true` と設定したコンテンツでのみ数式がレンダリングされる。

数式とマークダウンを共存させるために、数式はコードブロックとして記述するようにしている。
つまり、以下のような式を書くと
```
`$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}.$$`
```
以下のようにレンダリングされる。
`$$
p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}
$$`

---
以前 Academic テーマを使っていた時はこんなトリックを使わなくても上手くいっていた気がするので、もっとうまい方法があるかもしれない。