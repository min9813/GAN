### This repository is just for practicing Generative Adversary Nets

#### 雑です
- [PFNの実装](https://github.com/pfnet-research/chainer-gan-lib)を参考に、理解を深めることを目的に実装し直しています。PFNによる実装との相違点は、自分でbackward関数を作らないようにする点、conditional(wgan-gp、cramerのみ)を付け加えた点です。全てのGANにconditionalを付け加えていないのはconditionalの付け加え方が同じであり、簡単にできるからです。

##### 使い方
- train.pyに-m引数で使うGANの種類を選べます。
- 現在使えるGANはdcgan、wgan、wgan-gp、cramer-gan, be-gan, sn-gan, minibatch discrimination, feature matching, concidtional-ganです。

##### 全体的な気づき
- データは-1~1にリスケールさせる。これでハマった。
- 別ディレクトリ(hoge)内のモジュール(fuga.py)をインポートするとき、
```
import hoge
hoge.fuga
```
としていたが、これだとAttributeエラーが出る。
```
import hoge.fuga
```
としないといけない。

##### DCGANについて
-  cgan あり
- deconvolution は stride分各マス目間に０を埋める
- discriminatorの出力層をglobal average pooling にするとdiscriminatorの識別力が弱くて学習がうまく行かない
- 全結合層にして学習させるといい画像が得られた
- 正解画像は画像データを一周させたほうがいい結果が得られる
    - 実際は画像のスケールを255にしたほうが良い
- 損失関数はsigmoid, softmax, softplusどれにしてもあまり変わらなかった。
- Image fromarray や plt.imshow は (高さ, 幅, チャンネル)の順にしないといけない。


##### WGANについて
-  cgan なし
- clipping はdiscriminatorのみに行う
- generatorの最終出力層をtanhにしたのにもかかわらず、サンプル画像出力の際に、array*255のほうがmnistに関していい画像が得られる。
  - 255(array*0.5+0.5)にすると全体的に靄がかかって微妙。おそらく学習の際に、正解データが0〜1で入力されるため、generatorは出力を性にするように頑張っている？？？
  - それとも収束が遅いことが原因？

##### WGANGPについて
-  cgan あり
- [論文](https://arxiv.org/abs/1704.00028)には
<img src="https://latex.codecogs.com/gif.latex?$\alpha=0.0001,&space;\beta_1=0,&space;\beta_2=0.9$" title="$\alpha=0.0001, \beta_1=0, \beta_2=0.9$" />  
と書いてあったが、
<a href="https://github.com/igul222/improved_wgan_training">実装</a> では
<img src="https://latex.codecogs.com/gif.latex?$\alpha=0.0001,&space;\beta_1=0.5,&space;\beta_2=0.9$" title="$\alpha=0.0001, \beta_1=0.5, \beta_2=0.9$" />
となっていた。気をつけるべし。

- 収束が遅いのと、Cifarに対しては謎にwesserstein 距離が大きくなたりする。
  - 冒頭にも述べたように出力がF.tanhなのにデータのスケールが0~1になっていたことが原因
- エポックが200超えたらcriticとgeneratorを1:1で学習させたが無意味であった。
- F.sumで頑張って書くよりも、F.batch_l2_norm_squared使うほうが早い

- chainer-lib-gan都の違い
  - 乱数生成方法・batchデータのchainerへの変換は今のままで大丈夫、
  - unchainを外しても平気
    - unchainしたほうが速い。それはそう。
    - どっちにしても溜まるgradは変わらない
  - 問題はbackprop_funcを使うかどうか
    - 使うほうが過学習はしなくなったが溜まるgradは同じ出し計算結果も同じだし変わらないと思う。
  - image generateのときに設定したseedを戻し忘れてる。

##### Cramer ganについて
-  cgan あり
- discriminatorの出力は2次元以上にしないと、backward計算で直交ベクトルが出来上がり、勾配計算がnanとなりうまく行かなくなる
- backward はnet(x)とcritic(x)の組み合わせなので、critic(x)、net(x)の順にchainer.gradを適用すればよい。(自分でbackward関数作りたくないので)

##### began について
-  cgan なし
- discriminatorの中間層（隠れ層）の次元は128がいい。間違えて170としてしまったところ、変な出力となった。
- あえてAuto-Encoderにするメリットはなんだろう...ebganを見てみるしか？

##### Dragan について
-  cgan なし
- 実データの近傍付近に関する勾配のみ制限する、の理解ができた。
- 他と比べて綺麗なのかどうかはよくわからない。

##### snganについて
- cgan なし、実装予定
- 実行速度がwgan-gpの1.5倍くらいになった。
- 自分用のリンクの作り方が少しわかった。

##### improve techniqueについて
- minibatch_discrimination の実装が頭こんがらがった。[PFNの実装](https://github.com/pfnet-research/chainer-gan-lib)よりもreshapeをひとつ減らしています。
- feature matchingを付け加えた。これはsigmoid cross entropyを取る直前のものに用いた

##### CGANについて
- クラスラベルをone-hotの1チャンネルデータに変換している、3チャンネル画像の場合は3チャンネルにすべき？
  - 1チャンネルで学習はうまくできていた

##### まとめ
- 特になし

### generated images
##### cifar CGAN iteration 100000 by wgangp
![cifar CGAN iteration 100000 by wgangp](https://github.com/min9813/GAN/blob/master/sample_image/cgan_cifar/image_iteration:100000.png)
##### mnist CGAN iteration 100000 by dcgan
![mnist CGAN by dcgan](https://github.com/min9813/GAN/blob/master/sample_image/cgan_mnist/mnist_cgan.gif)
