"""
Configuration management for Explainable NLP Models.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the NLP model."""
    model_name: str = "distilbert-base-uncased"
    task: str = "sentiment-analysis"
    device: Optional[str] = None
    max_length: int = 512
    batch_size: int = 16


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: tuple = (12, 8)
    dpi: int = 300
    colormap: str = "Blues"
    save_plots: bool = True
    show_plots: bool = True


@dataclass
class LimeConfig:
    """Configuration for LIME explanations."""
    num_features: int = 10
    num_samples: int = 1000
    random_state: int = 42


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    visualization: VisualizationConfig
    lime: LimeConfig
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "outputs"


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_default_config()
        
        if Path(self.config_path).exists():
            self.load_config(self.config_path)
    
    def _load_default_config(self) -> AppConfig:
        """Load default configuration."""
        return AppConfig(
            model=ModelConfig(),
            visualization=VisualizationConfig(),
            lime=LimeConfig()
        )
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        self._update_config_from_dict(config_dict)
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'model' in config_dict:
            model_config = ModelConfig(**config_dict['model'])
            self.config.model = model_config
        
        if 'visualization' in config_dict:
            viz_config = VisualizationConfig(**config_dict['visualization'])
            self.config.visualization = viz_config
        
        if 'lime' in config_dict:
            lime_config = LimeConfig(**config_dict['lime'])
            self.config.lime = lime_config
        
        if 'log_level' in config_dict:
            self.config.log_level = config_dict['log_level']
        
        if 'data_dir' in config_dict:
            self.config.data_dir = config_dict['data_dir']
        
        if 'models_dir' in config_dict:
            self.config.models_dir = config_dict['models_dir']
        
        if 'output_dir' in config_dict:
            self.config.output_dir = config_dict['output_dir']
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")


# Global configuration instance
config_manager = ConfigManager()
