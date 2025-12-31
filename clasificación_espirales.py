import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_score
import time
import matplotlib.pyplot as plt


# Tomamos los dataset

path_morphology = r"C:/Users/Usuario/Documents/datasets de glaxias/morphology_catalogue.csv"
data_morph = pd.read_csv(path_morphology)

path_physical = r"C:/Users/Usuario/Documents/datasets de glaxias/useful_physical_measurements.parquet"
data_phys = pd.read_parquet(path_physical)


data = pd.merge(
    left= data_phys,
    right= data_morph,
    on= 'object_id',
    how= "inner"
)

print(data.shape)

## que variables nos interesan:

data = data[data['warning_galaxy_fails_training_cuts'] == False]




variables = ['ellipticity', 'log_kron_radius', 'log_segmentation_area', 'concentration', 'mumax_minus_mag_x', 'mumax_minus_mag_y', 'gini', 'asymmetry', 'smoothness', 'moment_20', 'sersic_sersic_vis_index', 'phz_pp_median_stellarmass', 'flux_detection_total_y', 'flux_vis_1fwhm_aper_y']
comparar = ['smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction']


#print(data.isna().mean().sort_values(ascending=False).head(50))

data_clean = data[variables + comparar].copy()



data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
data_clean = data_clean.dropna()

print(data_clean.shape)



conditions = [
    data_clean['smooth-or-featured_smooth_fraction'] > 0.7,
    data_clean['smooth-or-featured_featured-or-disk_fraction'] > 0.7,
    data_clean['merging_major-disturbance_fraction'] + data_clean['merging_merger_fraction'] > 0.4
]

choices = ['elliptical', 'spiral', 'disturbed']

data_clean['morph_class'] = np.select(conditions, choices, default='unknown')

df = data_clean[data_clean['morph_class'] != 'unknown']

print(df.shape)
print(df.head(6).T)






