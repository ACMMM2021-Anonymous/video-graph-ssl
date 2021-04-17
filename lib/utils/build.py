from .saver import Saver
from .summaries import TensorboardSummary

def creat_saver(cfg):
    saver = Saver(cfg)
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()

    return saver, writer