"""
Configuration module for AssetFormer.
Provides utilities to load and access decoding configurations.
"""
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TokenRanges:
    """Token range configuration for each type."""
    building_type: Dict[str, int]
    location_x: Dict[str, int]
    location_y: Dict[str, int]
    location_z: Dict[str, int]
    rotation_yaw: Dict[str, int]
    
    def get_range(self, token_type: str) -> tuple:
        """Get (start, end) range for a token type."""
        config = getattr(self, token_type)
        return config['start'], config['start'] + config['count']


@dataclass  
class DecodingConfig:
    """Configuration for decoding tokens to coordinates."""
    eos_id: int
    token_ranges: TokenRanges
    x_list: List[int]
    y_list: List[int]
    z_list: List[int]
    yaw_base: int
    yaw_step: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DecodingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        token_ranges = TokenRanges(
            building_type=config['token_ranges']['building_type'],
            location_x=config['token_ranges']['location_x'],
            location_y=config['token_ranges']['location_y'],
            location_z=config['token_ranges']['location_z'],
            rotation_yaw=config['token_ranges']['rotation_yaw'],
        )
        
        return cls(
            eos_id=config['tokens']['eos_id'],
            token_ranges=token_ranges,
            x_list=config['coordinates']['x_list'],
            y_list=config['coordinates']['y_list'],
            z_list=config['coordinates']['z_list'],
            yaw_base=config['rotation']['yaw_base'],
            yaw_step=config['rotation']['yaw_step'],
        )
    
    def get_coordinate_maps(self) -> tuple:
        """Generate coordinate mapping dictionaries."""
        xclass_x = {i: v for i, v in enumerate(self.x_list)}
        yclass_y = {i: v for i, v in enumerate(self.y_list)}
        zclass_z = {i: v for i, v in enumerate(self.z_list)}
        return xclass_x, yclass_y, zclass_z


# Default config path
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'decoding_config.yaml')
_config_cache = {}  # Cache configs by path


def get_decoding_config(config_path: str = None) -> DecodingConfig:
    """
    Get decoding configuration (with caching).
    
    Args:
        config_path: Path to config YAML file. If None, uses default config.
        
    Returns:
        DecodingConfig instance.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    
    # Normalize path for consistent caching
    config_path = os.path.abspath(config_path)
    
    if config_path not in _config_cache:
        _config_cache[config_path] = DecodingConfig.from_yaml(config_path)
    
    return _config_cache[config_path]
