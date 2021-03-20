---
title: "Hugo に外部コンテンツを良い感じで埋め込む"
date: 2021-03-20T08:12:54+09:00
draft: false
tags: ["hugo"]
---

# Shortcode を利用する方法
---

Hugo の [shortcode](https://gohugo.io/content-management/shortcodes/)を利用する方法。

## Youtube
{{</* youtube 2xkNJL4gJ9E */>}}
{{< youtube 2xkNJL4gJ9E>}}

## Twitter
{{</* twitter 1370289859995131905 */>}}
{{< twitter 1370289859995131905 >}}


# ブログカードを利用する方法
---

markdown 内に直接 html を書く。以下ははてなブログで使われているブログカードを利用する例。参考にしたサイトで例を作らせてもらった。

<iframe class="hatenablogcard" style="width:100%;height:155px;margin:15px 0;max-width:680px;" title="[Hugo]ブログカードを簡単に作成したい – Inomaso Blog" src="https://hatenablog-parts.com/embed?url=https://www.inomaso.com/post/2020/08/hugo-blogcard-generate/" frameborder="0" scrolling="no"></iframe>


なお Hugo のバージョンが0.60以上だと、記事内の html は無視されてしまうらしい。回避するために、`config.toml` に以下のように追記する。
```toml
[markup]
  [markup.goldmark]
    [markup.goldmark.renderer]
      unsafe = true
```
unsafe と書いてあるので後々問題が生じる可能性はある。
以下を参考にした。
<iframe class="hatenablogcard" style="width:100%;height:155px;margin:15px 0;max-width:680px;" title="Hugo v0.60以上を使うと、Markdown中のHTMLタグが「raw HTML omitted」となって消えてしまう - My External Storage" src="https://hatenablog-parts.com/embed?url=https://budougumi0617.github.io/2020/03/10/hugo-render-raw-html/" frameborder="0" scrolling="no"></iframe>

この方法だとはてなブログ側の仕様が変わった際に修正するのが大変そうなので、多用しない方が良いかもしれない。