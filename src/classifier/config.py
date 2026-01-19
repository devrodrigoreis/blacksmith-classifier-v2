
import yaml
import os
import logging

class Config:
    _instance = None

    def __new__(cls, config_path='config.yaml'):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
             # Fallback to looking in root if we are in a subdir? 
             # Or assume CWD is root.
             if os.path.exists(os.path.join('..', config_path)):
                 config_path = os.path.join('..', config_path)
             else:
                 logging.warning(f"Config file {config_path} not found. Using defaults.")
                 self.data = {}
                 return

        with open(config_path, 'r') as file:
            self.data = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def training(self):
        return self.data.get('training', {})

    @property
    def models(self):
        return self.data.get('models', {})

    @property
    def data_config(self):
        return self.data.get('data', {}) # 'data' is a dict method, so named data_config
    
    @property
    def memory(self):
        return self.data.get('memory', {})

# Global instance for easy access
config = Config()
