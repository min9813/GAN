### This repository is just for practicing Generative Adversary Nets

#### メモ

##### DCGANについて
- deconvolution は stride分各マス目間に０を埋める
- discriminatorの出力層をglobal average pooling にするとdiscriminatorの識別力が弱くて学習がうまく行かない
- 全結合層にして学習させるといい画像が得られた
- 正解画像は画像データを一周させたほうがいい結果が得られる
    - 実際は画像のスケールを255にしたほうが良い
- 損失関数はsigmoid, softmax, softplusどれにしてもあまり変わらなかった。
- Image fromarray や plt.imshow は (高さ, 幅, チャンネル)の順にしないといけない。

- ![Cifar epoch1](https://github.com/min9813/GAN/blob/master/sample_image/cifar/image_epoch_0001.png)
