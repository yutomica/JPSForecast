import optuna
import inspect
from optuna.samplers import BruteForceSampler

# class HydraBruteForceSampler(BruteForceSampler):
#     """
#     HydraのOptuna SweeperプラグインがTPESampler前提の引数を
#     強制的に渡してくるエラーを回避するためのラッパークラス。
#     """
#     def __init__(self, seed=None, **kwargs):
#         # TPE専用の不要な引数を捨てる
#         unsupported_keys = [
#             "consider_prior", 
#             "prior_weight", 
#             "multivariate", 
#             "group", 
#             "consider_endpoints", 
#             "n_startup_trials", 
#             "n_ei_candidates"
#         ]
#         for k in unsupported_keys:
#             kwargs.pop(k, None)
            
#         # BruteForceSamplerの初期化（seedのみ渡す）
#         super().__init__(seed=seed)

#     def after_trial(self, study, trial, state, values):
#         """
#         BruteForceSamplerは全探索が完了するとstudy.stop()を呼び出しますが、
#         Hydra環境下ではobjectiveの外で呼ばれるためRuntimeErrorになります。
#         Hydra側でn_trialsによってループを制御しているため、このエラーを回避します。
#         """
#         pass


class HydraBruteForceSampler(optuna.samplers.BruteForceSampler):
    """
    HydraのOptuna Sweeperから渡される不要な引数を自動的にフィルタリングし、
    BruteForceSamplerを正常に初期化するための堅牢なラッパー。
    """
    def __init__(self, *args, **kwargs):
        # 1. 親クラス (BruteForceSampler) の __init__ が受け取れる引数リストを取得
        sig = inspect.signature(optuna.samplers.BruteForceSampler.__init__)
        valid_params = sig.parameters.keys()

        # 2. 渡された kwargs のうち、有効なものだけを抽出（ホワイトリスト方式）
        # これにより 'consider_magic_clip' などの未知の引数を一括排除できる
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # 3. フィルタリング後の引数で初期化
        super().__init__(*args, **filtered_kwargs)

    def after_trial(self, study, trial, state, values):
        """
        BruteForceSamplerは全探索が完了するとstudy.stop()を呼び出しますが、
        Hydra環境下ではobjectiveの外で呼ばれるためRuntimeErrorになります。
        Hydra側でn_trialsによってループを制御しているため、このエラーを回避します。
        """
        pass