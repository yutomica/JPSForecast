# 基本的な学習実行コマンド
# 実装名ではなく、役割ベースのターゲット名を指定
python train.py +experiment=stacking domain=TAC target=tac_alpha

# Optunaによるハイパーパラメータ探索（Sweep）を行う場合
# python train.py -m +experiment=stacking mode=sweep domain=TAC target=tac_alpha
