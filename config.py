import numpy as np

particle_names = [
    'apo-ferritin',
    'beta-amylase',
    'beta-galactosidase',
    'ribosome',
    'thyroglobulin',
    'virus-like-particle'
]

particle_to_index = {
    'apo-ferritin': 0,
    'beta-amylase': 1,
    'beta-galactosidase': 2,
    'ribosome': 3,
    'thyroglobulin': 4,
    'virus-like-particle': 5
}

index_to_particle = {index: name for name, index in particle_to_index.items()}

particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}

particle_radius_blend = {
    'apo-ferritin': 65,
    'beta-amylase': 65,
    'beta-galactosidase': 95,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}

id_to_name = {
    1: "apo-ferritin", 
    2: "beta-amylase",
    3: "beta-galactosidase", 
    4: "ribosome", 
    5: "thyroglobulin", 
    6: "virus-like-particle"
}

BLOB_THRESHOLD = 255
CERTAINTY_THRESHOLD = 0.05
TRAIN_DATA_DIR = "./data/train_dataset"

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 11
tomo_type = "denoised"

classes = [1, 2, 3, 4, 5, 6] 