from torch.cuda.amp import GradScaler, autocast


class Scaler:
    def __init__(self, optimizer, use_fp16=False, *, set_to_none=False) -> None:
        self.optimizer = optimizer
        self.set_to_none = set_to_none
        self.autocast = autocast(enabled=use_fp16)
        self.scaler = GradScaler(enabled=use_fp16)

    def calculate_grad(self, loss):
        self.scaler.scale(loss).backward()

    def update_grad(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=self.set_to_none)

    def state_dict(self):
        r"""
        Returns the state of the scaler as a :class:`dict`.  It contains five entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor
        * ``"growth_interval"`` - a Python int containing the current growth interval
        * ``"_growth_tracker"`` - a Python int containing the number of recent consecutive unskipped steps.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler's state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        """
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        r"""
        Loads the scaler state.  If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        self.scaler.load_state_dict(state_dict)
