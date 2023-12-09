from typing import Optional, Dict
from pytorch_widedeep.callbacks import Callback, EarlyStopping as ES
import tabensemb
import numpy as np
import copy


class WideDeepCallback(Callback):
    def __init__(self, total_epoch, verbose):
        super(WideDeepCallback, self).__init__()
        self.val_ls = []
        self.total_epoch = total_epoch
        self.verbose = verbose

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict] = None,
        metric: Optional[float] = None,
    ):
        train_loss = logs["train_loss"]
        val_loss = logs["val_loss"]
        self.val_ls.append(val_loss)
        if epoch % tabensemb.setting["verbose_per_epoch"] == 0 and self.verbose:
            print(
                f"Epoch: {epoch + 1}/{self.total_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                f"Min val loss: {np.min(self.val_ls):.4f}"
            )


class EarlyStopping(ES):
    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        super(EarlyStopping, self).on_epoch_end(epoch=epoch, logs=logs, metric=metric)
        current = self.get_monitor_value(logs)
        # Detect loss anomaly
        if current > 1e8:
            self.stopped_epoch = epoch
            self.trainer.early_stop = True
