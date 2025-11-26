import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_score
import time
import matplotlib.pyplot as plt


#### DESCARGAMOS LOS DATASETS #####

path_morphology = r"C:/Users/Usuario/Desktop/cositas en python/clasificacion de galaxias/data/morphology_catalogue.csv"
data_morph = pd.read_csv(path_morphology)

path_physical = r"C:/Users/Usuario/Desktop/cositas en python/clasificacion de galaxias/data/useful_physical_measurements.parquet"
data_phys = pd.read_parquet(path_physical)

#Tomamos las variables que más nos interesan a la hora de inferir si se están fusionando o no y además, las variables de "merging"
variables_morph = ['object_id', 'mumax_minus_mag', 'mag_segmentation', 'ellipticity', 'kron_radius', 'smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction', 'cutout_width_arcsec',
                   'merging_none_fraction', 'merging_minor-disturbance_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction']
variables_phys = ['object_id', 'flux_detection_total', 'concentration', 'gini', 'asymmetry', 'moment_20', 'sersic_sersic_vis_index', 'phz_pp_median_redshift', 'phz_pp_median_stellarmass', 'phz_pp_median_sfr']

data = pd.merge(
    left= data_phys[variables_phys],
    right= data_morph[variables_morph],
    on= 'object_id',
    how= "inner"
)

# Existen valores infinitos y NaN, vamos a eliminarlos

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()

#print(data.shape) #(361913, 21)
#print(data.head(6).T)



####### CREAMOS EL CONJUNTO CON EL QUE INFERIREMOS SI LAS GALAXIAS SE ESTAN FUSIONANDO, (i.e. ELIMINAMOS LAS COLUMNAS "merging") #######

X = data.drop(['merging_none_fraction', 'merging_minor-disturbance_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction'], axis=1)

#print(X.shape)
#print(X.head(5).T)



###### CREAMOS EL OUTPUT: 1 SI HAY ALGUN INDICIO DE FUSIÓN, 0 DE LO CONTRARIO #######

data['Merging'] = 0

data['Merging'] = (data['merging_none_fraction'] < 0.5).astype(int)

y = data['Merging']



####################------------------------------------------##################


aux = pd.concat([X,y], axis=1)

corr = aux.corr()
correlaciones = corr.iloc[-1, :].sort_values()

print(correlaciones)














