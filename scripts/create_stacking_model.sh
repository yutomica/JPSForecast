# 基本的な学習実行コマンド
python train.py +experiment=stacking domain=TAC target.name=Future_Close_Tac

# Optunaによるハイパーパラメータ探索（Sweep）を行う場合
# python train.py -m +experiment=stacking mode=sweep domain=TAC target.name=Future_Close_Tac
