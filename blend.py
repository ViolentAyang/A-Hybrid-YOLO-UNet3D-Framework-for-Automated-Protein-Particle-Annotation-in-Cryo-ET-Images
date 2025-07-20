import pandas as pd
import numpy as np
from adaptive_dbscan import adaptive_blend_predictions

def blend_predictions(yolo_df, unet_df):
    """
    Enhanced blending function using adaptive DBSCAN clustering
    """
    return adaptive_blend_predictions(yolo_df, unet_df) 