from typing import Dict, Any


class ArgsParser(object):
    @staticmethod
    def get_or_error(key_value: Dict, key: str) -> Any:
        if key in key_value.keys():
            return key_value[key]
        else:
            raise KeyError(
                'Please provide a value for the parameter "%s"' % (key,))

    @staticmethod
    def get_or_default(key_value: Dict, key: str, default: Any) -> Any:
        if key in key_value.keys():
            return key_value[key]
        else:
            return default

    @staticmethod
    def get_or_default_and_remove(
            key_value: Dict, key: str, default: Any) -> Any:
        if key in key_value.keys():
            k = key_value[key]
            key_value.pop(key, None)
            return k
        else:
            return default
