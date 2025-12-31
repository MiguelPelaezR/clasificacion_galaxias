import numpy as np
import pandas as pd
import os
import tarfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_score
import time
import matplotlib.pyplot as plt
from PIL import Image
import io


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



# Generamos otro data frame solo con las variables que nos interesan

variables = ['ellipticity', 'log_kron_radius', 'log_segmentation_area', 'concentration', 'mumax_minus_mag_x', 'mumax_minus_mag_y', 'gini', 'asymmetry', 'smoothness', 'moment_20', 'sersic_sersic_vis_index', 'phz_pp_median_stellarmass', 'flux_detection_total_y', 'flux_vis_1fwhm_aper_y']
comparar = ['smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction', 'object_id', 'tile_index_y', 'jpg_loc_gz_arcsinh_vis_y']


#print(data.isna().mean().sort_values(ascending=False).head(50))

data_clean = data[variables + comparar].copy()


data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
data_clean = data_clean.dropna()

print(data_clean.shape)

######### clasificación de las galaxias

df = data_clean.copy()
df['morphology'] = 'none'

cond_eliptica = df['smooth-or-featured_smooth_fraction'] > 0.7
cond_espiral = df['smooth-or-featured_featured-or-disk_fraction'] > 0.7
cond_merging = df['merging_major-disturbance_fraction'] + df['merging_merger_fraction'] > 0.4

df.loc[cond_eliptica, 'morphology'] = 'elliptical'

df.loc[cond_espiral, 'morphology'] = 'spiral'

df.loc[cond_merging, 'morphology'] = 'merging'

df = df[df['morphology'] != 'none']

print(df.shape)


df_elliptical = df[df['morphology'] == 'elliptical'].copy()
df_spiral = df[df['morphology'] == 'spiral'].copy()
df_merging = df[df['morphology'] == 'merging'].copy()



########## GALLERY ########

tar_path = r'C:/Users/Usuario/Documents/datasets de glaxias/cutouts_jpg_gz_arcsinh_vis_y.tar'
tar = tarfile.open(tar_path, 'r')

# Creamos un diccionario: {nombre_archivo: objeto_miembro_del_tar}
tar_index = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}



def plot_samples_from_tar(df, n_samples=12):
    # Seleccionamos n galaxias al azar del dataframe
    samples = df.sample(n=n_samples)
    
    # Configuramos la cuadrícula (2 filas, 6 columnas)
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        ax = axes.flat[i]
        
        # Extraemos el nombre del archivo de la columna correspondiente
        full_path = row['jpg_loc_gz_arcsinh_vis_y']
        if pd.isna(full_path):
            ax.text(0.5, 0.5, "Sin ruta", ha='center')
            ax.axis('off')
            continue
            
        filename = os.path.basename(full_path)
        
        if filename in tar_index:
            # EXTRAER Y LEER DIRECTO DE MEMORIA
            member = tar_index[filename]
            file_content = tar.extractfile(member).read()
            img = Image.open(io.BytesIO(file_content))
            
            ax.imshow(img)
            # Ponemos el tile_index o el ID como título/texto
            ax.set_title(f"Tile: {row['tile_index_y']}", fontsize=8)
            ax.text(0.95, 0.05, f"ID: {row['object_id']}", 
                    color='white', transform=ax.transAxes, 
                    ha='right', fontsize=7, bbox=dict(facecolor='black', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No en TAR", ha='center', color='red')
        
        ax.axis('off')
    
    plt.show()


plot_samples_from_tar(df_elliptical, 12)

plot_samples_from_tar(df_spiral, 12)

plot_samples_from_tar(df_merging, 12)



