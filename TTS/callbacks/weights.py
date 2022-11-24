from trainer import Trainer


class WeightNormCallback:

    def __init__(self, interval: int = 200):
        self._interval = interval

    def __call__(self, trainer: Trainer, *args, **kwargs):
        if trainer.total_steps_done % self._interval != 0:
            return None

        for name, parameters in trainer.model.named_parameters():
            trainer.dashboard_logger.add_scalar(
                title=f"grad_norms/{name}",
                value=parameters.grad.data.norm(2).item(),
                step=trainer.total_steps_done
            )
            trainer.dashboard_logger.add_scalar(
                title=f"weight_norms/{name}",
                value=parameters.data.norm(2).item(),
                step=trainer.total_steps_done
            )

        return None
