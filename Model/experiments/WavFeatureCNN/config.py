# 此代码实现了自动编码器的配置管理

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Iterator

class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise ValueError(f"配置文件不存在: {config_path}")
        
        self.config_dir = self.config_path.parent
        self._config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 更新路径为绝对路径
            if 'data_dir' in config:
                config['data_dir'] = str(Path(config['data_dir']).resolve())
            if 'original_data_dir' in config:
                config['original_data_dir'] = str(Path(config['original_data_dir']).resolve())
            if 'save_dir' in config:
                config['save_dir'] = str(Path(config['save_dir']).resolve())
            
            # 确保必要的目录存在
            os.makedirs(config.get('save_dir', 'saved_models'), exist_ok=True)
            
            # 将配置保存为属性和字典
            self._config = config
            for key, value in config.items():
                setattr(self, key, value)
        
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        self._config.update(updates)
        for key, value in updates.items():
            setattr(self, key, value)
    
    def save(self, save_path: str = None) -> None:
        save_path = save_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
        except Exception as e:
            raise ValueError(f"保存配置文件失败: {str(e)}")
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._config)
    
    def __len__(self) -> int:
        return len(self._config)
    
    def items(self):
        return self._config.items()
    
    def keys(self):
        return self._config.keys()
    
    def values(self):
        return self._config.values()

def load_config(config_path: str = None) -> Config:
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    return Config(config_path) 