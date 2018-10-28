### This repository is just for practicing Generative Adversary Nets

#### 雑です

##### 使い方
- train.pyに-m引数で使うGANの種類を選べます。
- 現在使えるGANはDCGAN、WGAN、WGANGP、CGANです。

##### 全体的な気づき
- データは-1~1にリスケールさせる。これでハマった。

##### DCGANについて
- deconvolution は stride分各マス目間に０を埋める
- discriminatorの出力層をglobal average pooling にするとdiscriminatorの識別力が弱くて学習がうまく行かない
- 全結合層にして学習させるといい画像が得られた
- 正解画像は画像データを一周させたほうがいい結果が得られる
    - 実際は画像のスケールを255にしたほうが良い
- 損失関数はsigmoid, softmax, softplusどれにしてもあまり変わらなかった。
- Image fromarray や plt.imshow は (高さ, 幅, チャンネル)の順にしないといけない。


##### WGANについて
- clipping はdiscriminatorのみに行う
- generatorの最終出力層をtanhにしたのにもかかわらず、サンプル画像出力の際に、array*255のほうがmnistに関していい画像が得られる。
  - 255(array*0.5+0.5)にすると全体的に靄がかかって微妙。おそらく学習の際に、正解データが0〜1で入力されるため、generatorは出力を性にするように頑張っている？？？
  - それとも収束が遅いことが原因？

##### WGANGPについて
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


##### CGANについて
- クラスラベルをone-hotの1チャンネルデータに変換している、3チャンネル画像の場合は3チャンネルにすべき？
  - 1チャンネルで学習はうまくできていた

##### まとめ
- 今のところDCGANが一番きれい。エポック数が最大300と低いのが原因？

##### Cifar epoch100
![Cifar epoch1](https://github.com/min9813/GAN/blob/master/sample_image/cifar/image_epoch_0100.png)
