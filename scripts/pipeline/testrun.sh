set -e

# TCN Test
python train.py \
    model=tcn \
    domain=tac \
    data=sample \
    target=tac_alpha \
    period=tac_standard \
    cv=rrv \
    hparams=tcn/base,tcn/tac_alpha,anchor/tcn_tac_alpha \
    +hparams.max_epochs=2

# FT-Transformer Test
python train.py \
    model=ft_transformer \
    domain=tac \
    data=sample \
    target=tac_alpha \
    period=tac_standard \
    cv=rrv \
    hparams=ft_transformer/base,ft_transformer/tac_alpha,anchor/ft_transformer_tac_alpha \
    +hparams.max_epochs=2

# LightGBM Test
python train.py \
    model=lgbm \
    domain=tac \
    data=sample \
    target=tac_alpha \
    period=tac_standard \
    cv=rrv \
    hparams=lgbm/base,lgbm/tac_alpha,anchor/lgbm_tac_alpha \
    ++hparams.num_boost_round=10
