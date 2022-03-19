import os
from models.modules import shared_conv

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
