import pandas as pd
import tarfile
import os

# 1. Cargar datos
df_morph = pd.read_csv('data/morphology_catalogue.csv')

# Filtramos (por ejemplo, cogemos las primeras 5 espirales seguras para probar)
merging_minor = df_morph[df_morph['merging_major-disturbance_fraction'] > 0.4].head(5)

# 2. Configuración
tar_path = 'data/cutouts_jpg_gz_arcsinh_vis_y.tar' # El archivo de 3.8GB
output_dir = 'Imagenes_Galaxias'
os.makedirs(output_dir, exist_ok=True)


print(f"Abriendo el archivo TAR: {tar_path} (esto puede tardar unos segundos)...")

# 3. Abrir el TAR y buscar las imágenes
# Usamos 'r|' para leer stream si es muy grande, o 'r' normal.
with tarfile.open(tar_path, 'r') as tar:
    for index, row in merging_minor.iterrows():
        # El CSV nos da una ruta larga: /media/.../102042913_NEG...jpg
        # Solo nos interesa el nombre del archivo final: 102042913_NEG...jpg
        full_path = row['jpg_loc_gz_arcsinh_vis_y']
        if pd.isna(full_path):
            continue
            
        filename = os.path.basename(full_path) # Esto extrae solo el nombre.jpg
        
        # A veces dentro del TAR están en carpetas, a veces sueltos. 
        # Buscamos el archivo dentro del TAR que termine con ese nombre
        try:
            # Buscamos el miembro dentro del tar (esto es un truco rápido)
            # Nota: En archivos TAR muy grandes, buscar uno a uno puede ser lento.
            # Si sabes la estructura interna exacta es mejor, pero probemos buscando el nombre.
            
            # Opción rápida: Asumimos que el nombre en el tar es parecido
            member = None
            for m in tar.getmembers():
                if m.name.endswith(filename):
                    member = m
                    break
            
            if member:
                # Extraemos la imagen
                f = tar.extractfile(member)
                with open(f"{output_dir}/{filename}", 'wb') as out:
                    out.write(f.read())
                print(f"✅ Extraída: {filename}")
            else:
                print(f"❌ No encontrada en el TAR: {filename}")
                
        except Exception as e:
            print(f"Error con {filename}: {e}")

print("¡Proceso terminado!")




   