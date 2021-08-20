"""Contains basic configurations for models used in this project.

NOTE: Any new model should be registered in `MODEL_POOL` before using.
"""

import os.path

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = BASE_DIR + '/pretrain'

MODEL_POOL = {
    'stylegan_celebahq': {
        'tf_model_path':
            MODEL_DIR + '/karras2019stylegan-celebahq-1024x1024.pkl',
        'model_path': MODEL_DIR + '/stylegan_celebahq.pth',
        'gan_type': 'stylegan',
        'dataset_name': 'celebahq',
        'latent_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'min_val': -1.0,
        'max_val': 1.0,
        'output_channels': 3,
        'channel_order': 'RGB',
        'fused_scale': 'auto',
    },
    'stylegan_ffhq': {
        'tf_model_path': MODEL_DIR + '/karras2019stylegan-ffhq-1024x1024.pkl',
        'model_path': MODEL_DIR + '/stylegan_ffhq.pth',
        'gan_type': 'stylegan',
        'dataset_name': 'ffhq',
        'latent_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'min_val': -1.0,
        'max_val': 1.0,
        'output_channels': 3,
        'channel_order': 'RGB',
        'fused_scale': 'auto',
    },
}

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
STYLEGAN_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = False

MAX_IMAGES_ON_DEVICE = 4