import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typing import List, Tuple
from config import particle_names, particle_radius_blend

class AdaptiveDBSCAN:
    def __init__(self, base_confidence_threshold=0.5):
        self.base_confidence_threshold = base_confidence_threshold
        self.particle_radius = particle_radius_blend
        
    def calculate_adaptive_eps(self, coords: np.ndarray, particle_type: str) -> float:
        if len(coords) < 2:
            return self.particle_radius[particle_type]
        
        variance = np.var(coords, axis=0)
        mean_variance = np.mean(variance)
        
        base_radius = self.particle_radius[particle_type]
        
        density_factor = np.clip(mean_variance / 1000.0, 0.5, 2.0)
        adaptive_eps = base_radius * density_factor
        
        return adaptive_eps
    
    def calculate_confidence_weighted_min_samples(self, confidences: np.ndarray) -> int:
        if len(confidences) == 0:
            return 2
        
        mean_confidence = np.mean(confidences)
        confidence_factor = np.clip(1.0 - mean_confidence, 0.2, 0.8)
        
        base_min_samples = 2
        min_samples = max(1, int(base_min_samples * (1 + confidence_factor)))
        
        return min_samples
    
    def apply_confidence_weighted_averaging(self, cluster_points: pd.DataFrame) -> Tuple[float, float, float]:
        if 'confidence' in cluster_points.columns:
            weights = cluster_points['confidence'].values
            weights = weights / np.sum(weights)
            
            avg_x = np.average(cluster_points['x'], weights=weights)
            avg_y = np.average(cluster_points['y'], weights=weights)
            avg_z = np.average(cluster_points['z'], weights=weights)
        else:
            avg_x = cluster_points['x'].mean()
            avg_y = cluster_points['y'].mean()
            avg_z = cluster_points['z'].mean()
            
        return avg_x, avg_y, avg_z
    
    def filter_by_confidence_and_density(self, df: pd.DataFrame, particle_type: str) -> pd.DataFrame:
        if 'confidence' in df.columns:
            confidence_mask = df['confidence'] >= self.base_confidence_threshold
            df = df[confidence_mask]
        
        return df
    
    def cluster_predictions(self, df: pd.DataFrame, particle_type: str) -> pd.DataFrame:
        if len(df) < 2:
            return df
        
        df = self.filter_by_confidence_and_density(df, particle_type)
        
        if len(df) < 2:
            return df
        
        coords = df[['x', 'y', 'z']].values
        
        adaptive_eps = self.calculate_adaptive_eps(coords, particle_type)
        
        confidences = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))
        min_samples = self.calculate_confidence_weighted_min_samples(confidences)
        
        dbscan = DBSCAN(eps=adaptive_eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(coords)
        
        df = df.copy()
        df['cluster'] = labels
        
        clustered_results = []
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                noise_points = df[df['cluster'] == cluster_id].copy()
                if 'confidence' in noise_points.columns:
                    high_conf_noise = noise_points[noise_points['confidence'] > 0.7]
                    if len(high_conf_noise) > 0:
                        clustered_results.append(high_conf_noise)
                continue
            
            cluster_points = df[df['cluster'] == cluster_id].copy()
            
            avg_x, avg_y, avg_z = self.apply_confidence_weighted_averaging(cluster_points)
            
            cluster_points.loc[cluster_points.index[0], ['x', 'y', 'z']] = avg_x, avg_y, avg_z
            representative = cluster_points.iloc[[0]].copy()
            
            if 'confidence' in cluster_points.columns:
                representative.loc[representative.index[0], 'confidence'] = cluster_points['confidence'].mean()
            
            clustered_results.append(representative)
        
        if clustered_results:
            result_df = pd.concat(clustered_results, ignore_index=True)
            result_df = result_df.drop(columns=['cluster'], errors='ignore')
            return result_df
        else:
            return pd.DataFrame(columns=df.columns)

def adaptive_blend_predictions(yolo_df: pd.DataFrame, unet_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = pd.concat([yolo_df, unet_df], ignore_index=True)
    
    adaptive_dbscan = AdaptiveDBSCAN(base_confidence_threshold=0.3)
    
    final_results = []
    
    for particle_type in particle_names:
        particle_df = combined_df[combined_df['particle_type'] == particle_type].reset_index(drop=True)
        
        if len(particle_df) == 0:
            continue
        
        for experiment, group in particle_df.groupby('experiment'):
            group = group.reset_index(drop=True)
            
            clustered_group = adaptive_dbscan.cluster_predictions(group, particle_type)
            
            if len(clustered_group) > 0:
                final_results.append(clustered_group)
    
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)
        final_df = final_df.sort_values(by=['experiment', 'particle_type']).reset_index(drop=True)
        final_df['id'] = np.arange(len(final_df))
        return final_df
    else:
        return pd.DataFrame()