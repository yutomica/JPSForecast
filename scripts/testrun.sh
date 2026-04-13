
set -e

# python train.py \
#     model=tcn \
#     domain=tac \
#     data=sample \
#     target=tac_vol_scaled_asym_return \
#     period=tac_standard \
#     cv=rrv \
#     ++hparams.objective="asymmetric_mse" \
#     +hparams.max_epochs=2

# python train.py \
#     model=tcn \
#     domain=tac \
#     data=sample \
#     target=tac_max_neg_path \
#     period=tac_standard \
#     cv=rrv \
#     ++hparams.objective="quantile" \
#     ++hparams.alpha=0.1 \
#     +hparams.max_epochs=2

# python train.py \
#     model=tcn \
#     domain=str \
#     data=sample \
#     target=str_sharpe_adj \
#     period=str_standard \
#     cv=rrv \
#     ++hparams.objective="fair" \
#     ++hparams.fair_c=10.0 \
#     +hparams.max_epochs=2

# python train.py \
#     model=tcn \
#     domain=str \
#     data=sample \
#     target=str_mdd \
#     period=str_standard \
#     cv=rrv \
#     ++hparams.objective="tweedie" \
#     ++hparams.tweedie_variance_power=1.2 \
#     +hparams.max_epochs=2

python train.py \
    model=ft_transformer \
    domain=tac \
    data=sample \
    target=tac_vol_scaled_asym_return \
    period=tac_standard \
    cv=rrv \
    ++hparams.objective="asymmetric_mse" \
    +hparams.max_epochs=2

python train.py \
    model=ft_transformer \
    domain=tac \
    data=sample \
    target=tac_max_neg_path \
    period=tac_standard \
    cv=rrv \
    ++hparams.objective="quantile" \
    ++hparams.alpha=0.1 \
    +hparams.max_epochs=2

python train.py \
    model=ft_transformer \
    domain=tac \
    data=sample \
    target=str_sharpe_adj \
    period=tac_starndard \
    cv=rrv \
    ++hparams.objective="fair" \
    ++hparams.fair_c=10.0 \
    +hparams.max_epochs=2

python train.py \
    model=ft_transformer \
    domain=tac \
    data=sample \
    target=str_mdd \
    period=tac_standard \
    cv=rrv \
    ++hparams.objective="tweedie" \
    ++hparams.tweedie_variance_power=1.2 \
    +hparams.max_epochs=2
