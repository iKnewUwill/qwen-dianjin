from transformers import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3PRMConfig(Qwen3Config):

    model_type = "qwen3"

    def __init__(
        self,
        alpha=1.0,
        num_labels=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_labels = num_labels
