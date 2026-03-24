from optuna.samplers import BruteForceSampler

class HydraBruteForceSampler(BruteForceSampler):
    """
    HydraのOptuna SweeperプラグインがTPESampler前提の引数を
    強制的に渡してくるエラーを回避するためのラッパークラス。
    """
    def __init__(self, seed=None, **kwargs):
        # TPE専用の不要な引数を捨てる
        unsupported_keys = [
            "consider_prior", 
            "prior_weight", 
            "multivariate", 
            "group", 
            "consider_endpoints", 
            "n_startup_trials", 
            "n_ei_candidates"
        ]
        for k in unsupported_keys:
            kwargs.pop(k, None)
            
        # BruteForceSamplerの初期化（seedのみ渡す）
        super().__init__(seed=seed)

    def after_trial(self, study, trial, state, values):
        """
        BruteForceSamplerは全探索が完了するとstudy.stop()を呼び出しますが、
        Hydra環境下ではobjectiveの外で呼ばれるためRuntimeErrorになります。
        Hydra側でn_trialsによってループを制御しているため、このエラーを回避します。
        """
        pass