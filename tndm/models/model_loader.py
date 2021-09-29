from __future__ import annotations
from collections import defaultdict
import json
from abc import ABC, abstractmethod, abstractstaticmethod
import os
from typing import Dict, Any

from tndm.parser import Parser
from tndm.utils import CustomEncoder, upsert_empty_folder


class ModelLoader(ABC):
    
    @staticmethod
    def load(location, model_class) -> ModelLoader:
        with open(os.path.join(location, 'settings.json'), 'r') as fp:
            settings = json.load(fp)
        
        model_settings, layers_settings = Parser.parse_model_settings(settings)
        model = model_class(**model_settings, 
            layers=layers_settings)

        model.load_weights(os.path.join(location, 'weights'))
        return model

    @abstractmethod
    def get_settings(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def save(self, location):
        settings = self.get_settings()
        upsert_empty_folder(location)
        with open(os.path.join(location, 'settings.json'), 'w') as fp:
            json.dump(settings, fp, cls=CustomEncoder)
        self.save_weights(os.path.join(location, 'weights'))
