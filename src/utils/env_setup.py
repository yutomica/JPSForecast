import os
import logging

def setup_environment():
    """
    並列処理時のスレッド競合（オーバーサブスクリプション）を防ぐための環境変数設定。
    各種ライブラリ（numpy, pandas, torchなど）がインポートされる前に呼び出す必要があります。
    """
    # TCNの並列学習時に各プロセスがCPUを過剰に使用するのを防ぎ、パフォーマンスを安定させる
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["ACCELERATE_NUM_THREADS"] = "1"
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # PyTorch内部の計算エラーを回避
    os.environ["GCD_ALL_ACTIVE_PROCESSORS"] = "1" # Apple Grand Central Dispatch のスレッド展開を制限
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE" # CPUのビジーウェイト（スピンロック）を防止

    # PyTorchの内部スレッドプールの制限
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    # PyArrowの独自スレッドプールの制限
    try:
        import pyarrow as pa
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except ImportError:
        pass

    # Zarr/Bloscの独自スレッドプールの制限
    try:
        import numcodecs
        numcodecs.blosc.set_nthreads(1)
        numcodecs.blosc.use_threads = False
    except ImportError:
        pass

    # MLflow周辺の冗長なログ・警告を抑制する。
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    mlflow_models_logger = logging.getLogger("mlflow.models.model")
    mlflow_pyfunc_logger = logging.getLogger("mlflow.pyfunc")
    logging_state = {
        "mlflow_models_logger": mlflow_models_logger,
        "mlflow_pyfunc_logger": mlflow_pyfunc_logger,
        "prev_models_level": mlflow_models_logger.level,
        "prev_pyfunc_level": mlflow_pyfunc_logger.level,
    }
    mlflow_models_logger.setLevel(logging.ERROR)
    mlflow_pyfunc_logger.setLevel(logging.ERROR)

    return logging_state
