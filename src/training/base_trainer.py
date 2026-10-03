import os
import logging
import torch
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config, log_filename='training.log'):
        # Scripts that only load a checkpoint pass their own log name, so they
        # do not append to the log of the run that produced it
        self.config = config
        self.log_filename = log_filename
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Setup logging configuration"""
        log_path = os.path.join(self.config.model_output_dir, self.log_filename)
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
        # Created by scripts/evaluate.py when it first writes, so training runs
        # do not leave an empty directory behind
        self.evaluation_dir = os.path.join(self.config.model_output_dir, 'evaluation')

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
