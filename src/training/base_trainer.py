import os
import logging
import torch
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Setup logging configuration"""
        log_path = os.path.join(self.config.model_output_dir, 'training.log')
        # force=True because importing src.data_loader already called
        # basicConfig, which would otherwise make this a silent no-op and leave
        # training.log empty
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create output directories"""
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.config.model_output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @abstractmethod
    def setup_data(self):
        """Setup data loaders - to be implemented by subclasses"""
        pass

    @abstractmethod
    def setup_model(self):
        """Setup model - to be implemented by subclasses"""
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        """Train for one epoch - to be implemented by subclasses"""
        pass

    @abstractmethod
    def validate(self, epoch):
        """Validate model - to be implemented by subclasses"""
        pass
