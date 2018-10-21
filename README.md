### This repository is just for practicing Generative Adversary Nets

#### 雑です

##### 使い方
- train.pyに-m引数で使うGANの種類を選べます。
- 現在使えるGANはDCGAN、WGAN、WGANGPです。

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
- 収束が遅いのと、Cifarに対しては謎にwesserstein 距離が大きくなたりする。

##### Cifar epoch100
![Cifar epoch1](https://github.com/min9813/GAN/blob/master/sample_image/cifar/image_epoch_0100.png)
