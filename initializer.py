import threading
from initial import initialize_models, get_target_dict_internal

_initialized = False
_initialize_lock = threading.Lock()

def initialize_once():
    global _initialized
    with _initialize_lock:
        if not _initialized:
            initialize_models()
            _initialized = True

def get_target_dict():
    initialize_once()
    return get_target_dict_internal()
