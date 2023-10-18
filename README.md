# AHC025 {ignore=true}

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [10/14](#1014)
  - [問題を見る](#問題を見る)
  - [一投目](#一投目)
- [10/15](#1015)
  - [二投目](#二投目)
  - [三投目](#三投目)
  - [比較回数を最小化するソート](#比較回数を最小化するソート)
  - [四投目](#四投目)
- [10/16](#1016)
  - [~七投目](#七投目)
  - [八投目](#八投目)
- [10/17](#1017)
  - [11投目](#11投目)
  - [解法選択に logistic regression](#解法選択に-logistic-regression)
  - [12投目](#12投目)
- [10/18](#1018)
  - [number partitioning problem と longest processing time heuristics](#number-partitioning-problem-と-longest-processing-time-heuristics)
  - [14投目](#14投目)

<!-- /code_chunk_output -->

---

## 10/14

### 問題を見る

推定系が出て頭を抱える

### 一投目

* **クエリ:**
  * 全アイテムからなる集合を二分割して天秤に乗せる操作を Q 回繰り返す
  * 条件が Q 個得られる
* **重み推定:**
  * 入力生成方法に従って重み配列をランダム生成することを繰り返し、満たす条件数が多くなるよう山登りする
* **分割:**
  * 得られた重み配列を D グループに分配 (mod D で)
  * 山登りで等分
    * 1 アイテムを他グループに移動
    * 2 つのグループから各 1 つずつアイテムを選び、交換
* **提出:**
  * https://atcoder.jp/contests/ahc025/submissions/46516234
  * 2290075965

---

## 10/15

### 二投目

* **クエリ:** [一投目](#一投目)と同様
* **重み推定:**
  * 入力生成方法に従って重み配列をランダム生成して初期解とする
  * 満たす条件数が多くなるよう、アイテムの重みを一点変更・二点交換で山登りする
* **分割:**: [一投目](#一投目)と同様
* **提出:**
  * https://atcoder.jp/contests/ahc025/submissions/46593907
  * 1179176525

### 三投目

* **クエリ:**
  * 全アイテムからなる集合を二分割して天秤に乗せる操作を Q 回繰り返す
  * 二分割のサイズに -2~2 のゆらぎを持たせる
* **重み推定:** [二投目](#二投目)と同様
* **分割:**: [一投目](#一投目)と同様
* **提出:**
  * https://atcoder.jp/contests/ahc025/submissions/46596978
  * 1131091655

---

### 比較回数を最小化するソート

色々調べると [Merge-insertion sort (Ford-Johnson algorithm)](https://en.wikipedia.org/wiki/Merge-insertion_sort) が比較回数において優れていることがわかる

* [A001768](https://oeis.org/A001768): Sorting numbers: number of comparisons for merge insertion sort of n elements.
* [A036604](https://oeis.org/A036604): Sorting numbers: minimal number of comparisons needed to sort n elements.
  * 最適解: 先頭 15 項までしか乗っていない

取り回しの効く実装が Web に落ちていなかったので、自前で実装したら時間がかかってしまった

下記に 21 要素のソート例が載っており、参考になった
* Knuth, Donald E. (1998), "Merge insertion", The Art of Computer Programming, Vol. 3: Sorting and Searching (2nd ed.), pp. 184–186

---

### 四投目

* `A001768(N) <= Q` の場合
  * Ford-Johnson algorithm で N アイテムを並び替える
  * 残りのターンで[三投目](#三投目)同様のクエリを投げる
  * 全実行時間の 75% を使って以下を実行
    * 入力生成方法に従って**ソートされた**重み配列をランダム生成することを繰り返す
    * 満たす条件が多くなるように山登り
  * 分割方法は[三投目](#三投目)と同様
* それ以外
  * [三投目](#三投目)と同様
* **提出:**
  * https://atcoder.jp/contests/ahc025/submissions/46616705
  * 622417139

---

## 10/16

### ~七投目

ソートされた重み配列の山登りを工夫したり焼きなましに変更したりしてみたが、効果は薄い

### 八投目

クエリ数 Q の制約が厳しいときは、`A001768(K) <= Q`を満たす最大の K を探して、N 個のアイテムを予め K 個に分割してからソートすればよい

* **提出**
  * https://atcoder.jp/contests/ahc025/submissions/46650346
  * 303664289

---

## 10/17

### 11投目

* `A001768(K) <= Q * 3 / 5 && K <= N` を満たす最大の K を決めて、アイテムを K 分割してソート
* D 個のグループにソート済みの要素を**降順・蛇腹状**に分配する
* N, D, Q の値に応じて、以下の 2 つのアルゴリズムのよい方を選択する
  1. 2 グループを選択して重みを比較し、重い方に含まれる最小要素を軽い方に移動させることを繰り返す
  2. D グループの重みを Merge-insertion sort で比較し、最大重みグループの最小要素を最小重みグループに移動させることを繰り返す
* **提出**:
  * https://atcoder.jp/contests/ahc025/submissions/46671830
  * 231362559

### 解法選択に logistic regression

2 アルゴリズムの選択には 10000 ケースの実行結果を用いてロジスティック回帰を使ってみた
* https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569
* https://blog.amedama.jp/entry/2018/08/26/095444
* [てきとうコード](scripts/logistic_regression.py)

相変わらず何もわからないが、スコアが上がったのでヨシ

---

### 12投目

D 個のグループにソート済みの要素を**降順・蛇腹状**に分配していたが、これをもう少しマシにする
* 入力生成方法に従って K 要素の**ソート済み**重み配列を得ることを繰り返し、その平均値を暫定的な重みとする
* 暫定的な重みに基づいてグループへの分配を山登りで最適化し、得られた分配を初期状態とする
* 初期状態から[11投目](#11投目)で述べた 2 つのアルゴリズムを実行する
* **提出:**
  * https://atcoder.jp/contests/ahc025/submissions/46674536
  * 194475449

---

## 10/18

### number partitioning problem と longest processing time heuristics

重みが陽にわかる場合は[数分割問題(number partitioning probrem)](https://scmopt.github.io/opt100/76npp.html) の分割が 3 以上のケースに該当する

[複数装置スケジューリング問題](https://scmopt.github.io/opt100/76npp.html#%E8%A4%87%E6%95%B0%E8%A3%85%E7%BD%AE%E3%82%B9%E3%82%B1%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0%E5%95%8F%E9%A1%8C)と見なすことができて、**longest processing time (LPT) heuristics** を用いることでよい近似解を得られるらしい

重みの大きい要素から順に重みの和が最小のグループに振り分けることを繰り返すだけで、今回の問題と相性が良さそう

思えば、**降順・蛇腹状**の分配が良いスコアを出したのは LPT heuristics と序盤の分配が似ているからかもしれない

---

### 14投目

Merge-insertion sort した K 要素を LPT heuristics に従いグループに分配していく

重み最小グループの判定は高々 K-1 回の比較で行えるが、比較回数の節約のためにグループの大小関係を DAG で保持しておく（DAG は小->大に向けて辺を張る）

* 要素を降順にキューに詰める
* 最初の D 要素をキューから取り出し、各グループに分配する
  * 比較は必要なく、DAG はパスグラフになる
* 以下を繰り返す
  * DAG は入次数 0 のノード(=最小重みグループ)を 1 つだけ持つので、これを root とする
  * キューから先頭要素を取り出し、root に分配する
  * キューが空なら余計な比較をしないよう break する
  * root ノードの重さが不明になったので、root ノードから出る辺を削除する
  * 入次数 0 のノードの大小を比較して、小 -> 大に辺を張る
    * 森をマージしていくイメージ
    * 入次数 0 のノードの個数を M とすれば、比較は高々 M-1 回

N, D に対して実験的に比較回数の上限 `cmp[N][D]` を求めておき、`A001768(K) + cmp[K][D] <= Q && K <= N` を満たす最大の K に対して上記アルゴリズムを実行して解を得る

* **提出**:
  * https://atcoder.jp/contests/ahc025/submissions/46694346
  * 117702800

この時点で 45 位 90G くらい、ようやくスタートラインに立った気がする