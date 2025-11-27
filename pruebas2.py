
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

print('haciendo cambios')

####### CREAMOS EL CONJUNTO CON EL QUE INFERIREMOS SI LAS GALAXIAS SE ESTAN FUSIONANDO, (i.e. ELIMINAMOS LAS COLUMNAS "merging") #######

X = data.drop(['merging_none_fraction', 'merging_minor-disturbance_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction'], axis=1)

#print(X.shape)
#print(X.head(5).T)



###### CREAMOS EL OUTPUT: 1 SI HAY ALGUN INDICIO DE FUSIÓN, 0 DE LO CONTRARIO #######

data['Merging'] = 0

data['Merging'] = (data['merging_none_fraction'] < data['merging_merger_fraction'] + data['merging_major-disturbance_fraction']).astype(int)

y = data['Merging']
print(y.value_counts())


'''
####################------------------------------------------##################


aux = pd.concat([X,y], axis=1)

corr = aux.corr()
correlaciones = corr.iloc[-1, :].sort_values()

print(correlaciones)
'''





#### PREPARAR LOS DATOS PARA ENTRENAMIENTO Y PRUEBA, ADEMÁS DE ESTANDARIZAR LOS DATOS ##########

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify= y, random_state=42)

## Estandarizamos las muestras:
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns, index= X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns= X_test.columns, index= X_test.index)


######## CREAMOS LOS MODELOS DONDE VAMOS A IMPLEMENTAR LOS DATOS #######

## XGBoosting
xgb = XGBClassifier(n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    scale_pos_weight=7, 
                    objective='binary:logistic',
                    n_jobs=-1,
                    eval_metric='auc'
                    )


## Random Forest
rf = RandomForestClassifier(
    n_estimators= 200,
    max_depth= 8,
    class_weight= 'balanced',
    n_jobs= -1
)


####### ENTRENAMIENTO Y PREDICIONES DE LOS MODELOS ######

## Entrenamiento
start_time_xgb = time.time()
xgb.fit(X_train, y_train)


#y_pred_xgb = xgb.predict(X_test_scaled)
y_probs_xgb = xgb.predict_proba(X_test_scaled)[:, 1] 
y_pred_xgb_ajusted = (y_probs_xgb >= 0.6).astype(int)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

#---------------------------------#

start_time_rf = time.time()
rf.fit(X_train, y_train)

#y_pred_rf = rf.predict(X_test_scaled)
y_probs_rf = rf.predict_proba(X_test_scaled)[:, 1] 
y_pred_rf_ajusted = (y_probs_rf >= 0.5).astype(int)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

print(f'Tiempo de entrenamiento y predicción de XGBoosting: {xgb_train_time} segundos')
print(f'Tiempo de entrenamiento y prendicción de Random Forest: {rf_train_time} segundos')


######## EVALUACIÓN DE LOS MODELOS ########

print('\n~~~~~~~~~~~~~~ EVALUACION DE XGBoosting ~~~~~~~~~~~~~~')
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_xgb_ajusted):.4f}")
print(f'F1-score:  {f1_score(y_test, y_pred_xgb_ajusted):.4f}')
print(f'Recall:   {recall_score(y_test, y_pred_xgb_ajusted):.4f}')
print(f'Precision:   {precision_score(y_test, y_pred_xgb_ajusted):.4f}')
print('Confusion matrix:\n')
print(confusion_matrix(y_test, y_pred_xgb_ajusted))

print('\n~~~~~~~~~~~~~~ EVALUACION DE Random Forest ~~~~~~~~~~~~~~')
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_rf_ajusted):.4f}")
print(f'F1-score:  {f1_score(y_test, y_pred_rf_ajusted):.4f}')
print(f'Recall:   {recall_score(y_test, y_pred_rf_ajusted):.4f}')
print(f'Precision:   {precision_score(y_test, y_pred_rf_ajusted):.4f}')
print('Confusion matrix:\n')
print(confusion_matrix(y_test, y_pred_rf_ajusted))













