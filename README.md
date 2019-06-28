
PFN intern 2019 の [Graph neural network フルスクラッチ実装課題](https://github.com/pfnet/intern-coding-tasks/tree/master/2019/machine_learning) が面白そうだったので実装してみました．

下記コマンドを実行すると学習が始まります．
```
python trainer.py
```

以下のパラメータで検証データに対する平均Accuracyが60%程度になることを確認しました．
- 集約ステップ数: 2
- 特徴ベクトルの次元数: 8
- Epochs: 50
- batch size: 32
- Optimizer: Adam
  - lr: 0.001
  - beta1: 0.9
  - beta2: 0.999
  - eps: 1e-8
