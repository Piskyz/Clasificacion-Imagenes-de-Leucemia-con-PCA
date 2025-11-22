"""Carga imagenes 500

# Carga de imagenes 500 de cada conjunto de datos

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

"""# carga de imagenes originales cortado a 500

Testing data original
"""

#Path
testing_data_path = "data/testing_data/C-NMC_test_final_phase_data/"

#Lista imagenes
testing_data = []

# Cargar solo las primeras 500 imágenes válidas
max_images = 500
for i in range(1, 2587):
    if len(testing_data) >= max_images:
        break
    nombre_archivo = os.path.join(testing_data_path, f"{i}.bmp")
    if not os.path.exists(nombre_archivo):
        print(f"Advertencia: no existe {nombre_archivo}")
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        testing_data.append(img)
        if len(testing_data) % 50 == 0:
            print(f"Cargadas {len(testing_data)}/{max_images} imágenes")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes cargadas: {len(testing_data)}")

# Mostrar las primeras N imágenes cargadas en testing_data
N = 6
if isinstance(testing_data, list) and len(testing_data) > 0:
    n = min(N, len(testing_data))
    plt.figure(figsize=(3*n, 4))
    for i in range(n):
        img = testing_data[i]
        if img is None:
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax = plt.subplot(1, n, i+1)
        plt.imshow(disp, cmap=cmap)
        plt.title(f"#{i+1}  {disp.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("testing_data no es una lista o está vacía. Asegúrate de haber ejecutado la celda que carga las imágenes.")

"""Training data fold 0"""

# path
training_data_fold_0_path = "data/training_data/fold_0/all/"

#Lista imagenes
training_data_fold_0 = []

#Definicion de parametros
total_loaded = 0
max_images = 500  # límite de imágenes a cargar

max_x = 52
max_y = 37
max_z = 10

#ciclo para carga de imagenes
for x in range(1, max_x + 1):
    if total_loaded >= max_images:
        break
    found_any_for_x = False
    for y in range(1, max_y + 1):
        if total_loaded >= max_images:
            break
        found_any_for_y = False
        for z in range(1, max_z + 1):
            if total_loaded >= max_images:
                break
            filename = f"UID_{x}_{y}_{z}_all.bmp"
            filepath = os.path.join(training_data_fold_0_path, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    training_data_fold_0.append(img)
                    total_loaded += 1
                    found_any_for_y = True
                    found_any_for_x = True
                    if total_loaded % 50 == 0 or total_loaded == max_images:
                        print(f"Cargado: {filename} (Total cargados: {total_loaded}/{max_images})")
                else:
                    print(f"Advertencia: no se pudo cargar {filename}")
            # No archivo = continuar para siguiente z
        # Si no se encontró ninguna imagen para este y (en cualquiera de los z), continuar al siguiente y
        if not found_any_for_y:
            continue
    # Si no se encontró ninguna imagen para este x (en cualquiera de los y,z) romper loop y pasar al siguiente x
    if not found_any_for_x:
        print(f"No se encontraron imágenes para x={x}, saltando al siguiente valor de x.")
        continue

print(f"Total imágenes cargadas: {total_loaded}")

#impresion de primeras 6 imagenes para verq ue se carguen de buena manera

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i in range(min(6, len(training_data_fold_0))):
    ax = axes[i // 3, i % 3]
    # Convertir BGR a RGB para mostrar correctamente con matplotlib
    img_rgb = cv2.cvtColor(training_data_fold_0[i], cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"Imagen {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

"""Training data fold 1"""

# path
training_data_fold_1_path = "data/training_data/fold_1/all/"

#Lista imagenes
training_data_fold_1 = []

#Definicion de parametros
total_loaded = 0
max_images = 500  # límite de imágenes a cargar

max_x = 52
max_y = 181
max_z = 15

for x in range(1, max_x + 1):
    if total_loaded >= max_images:
        break
    found_any_for_x = False
    for y in range(1, max_y + 1):
        if total_loaded >= max_images:
            break
        found_any_for_y = False
        for z in range(1, max_z + 1):
            if total_loaded >= max_images:
                break
            filename = f"UID_{x}_{y}_{z}_all.bmp"
            filepath = os.path.join(training_data_fold_1_path, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    training_data_fold_1.append(img)
                    total_loaded += 1
                    found_any_for_y = True
                    found_any_for_x = True
                    if total_loaded % 50 == 0 or total_loaded == max_images:
                        print(f"Cargado: {filename} (Total cargados: {total_loaded}/{max_images})")
                else:
                    print(f"Advertencia: no se pudo cargar {filename}")
            # No archivo = continuar para siguiente z
        # Si no se encontró ninguna imagen para este y (en cualquiera de los z), continuar al siguiente y
        if not found_any_for_y:
            continue
    # Si no se encontró ninguna imagen para este x (en cualquiera de los y,z) romper loop y pasar al siguiente x
    if not found_any_for_x:
        print(f"No se encontraron imágenes para x={x}, saltando al siguiente valor de x.")
        continue

print(f"Total imágenes cargadas: {total_loaded}")

#impresion de primeras 6 imagenes para ver que se carguen de buena manera

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i in range(min(6, len(training_data_fold_1))):
    ax = axes[i // 3, i % 3]
    # Convertir BGR a RGB para mostrar correctamente con matplotlib
    img_rgb = cv2.cvtColor(training_data_fold_1[i], cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"Imagen {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

"""Training data fold 2"""

# path
training_data_fold_2_path = "data/training_data/fold_2/all/"

#Lista imagenes
training_data_fold_2 = []

#Definicion de parametros
total_loaded = 0
max_images = 500  # límite de imágenes a cargar

max_x = 79
max_y = 40
max_z = 15

for x in range(1, max_x + 1):
    if total_loaded >= max_images:
        break
    found_any_for_x = False
    for y in range(1, max_y + 1):
        if total_loaded >= max_images:
            break
        found_any_for_y = False
        for z in range(1, max_z + 1):
            if total_loaded >= max_images:
                break
            filename = f"UID_{x}_{y}_{z}_all.bmp"
            filepath = os.path.join(training_data_fold_2_path, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    training_data_fold_2.append(img)
                    total_loaded += 1
                    found_any_for_y = True
                    found_any_for_x = True
                    if total_loaded % 50 == 0 or total_loaded == max_images:
                        print(f"Cargado: {filename} (Total cargados: {total_loaded}/{max_images})")
                else:
                    print(f"Advertencia: no se pudo cargar {filename}")
            # No archivo = continuar para siguiente z
        # Si no se encontró ninguna imagen para este y (en cualquiera de los z), continuar al siguiente y
        if not found_any_for_y:
            continue
    # Si no se encontró ninguna imagen para este x (en cualquiera de los y,z) romper loop y pasar al siguiente x
    if not found_any_for_x:
        print(f"No se encontraron imágenes para x={x}, saltando al siguiente valor de x.")
        continue

print(f"Total imágenes cargadas: {total_loaded}")

#impresion de primeras 6 imagenes para ver que se carguen de buena manera
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i in range(min(6, len(training_data_fold_2))):
    ax = axes[i // 3, i % 3]
    # Convertir BGR a RGB para mostrar correctamente con matplotlib
    img_rgb = cv2.cvtColor(training_data_fold_2[i], cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"Imagen {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

"""Validation data"""

#Path
validation_data_path = "data/validation_data/C-NMC_test_prelim_phase_data/"

#Lista imagenes
validation_data = []

# Cargar máximo 500 imágenes válidas
max_images = 500
for i in range(1, 1867):
    if len(validation_data) >= max_images:
        break
    nombre_archivo = os.path.join(validation_data_path, f"{i}.bmp")
    if not os.path.exists(nombre_archivo):
        print(f"Advertencia: no existe {nombre_archivo}")
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        validation_data.append(img)
        if len(validation_data) % 50 == 0:
            print(f"Cargadas {len(validation_data)}/{max_images} imágenes")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes cargadas: {len(validation_data)}")

# Mostrar las primeras N imágenes cargadas en testing_data
N = 6
if isinstance(validation_data, list) and len(validation_data) > 0:
    n = min(N, len(validation_data))
    plt.figure(figsize=(3*n, 4))
    for i in range(n):
        img = validation_data[i]
        if img is None:
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax = plt.subplot(1, n, i+1)
        plt.imshow(disp, cmap=cmap)
        plt.title(f"#{i+1}  {disp.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("testing_data no es una lista o está vacía. Asegúrate de haber ejecutado la celda que carga las imágenes.")

"""# Carga de las imagenes reconstruidas

testing data
"""

#Path (versión reconstruida)
testing_data_reconstruida_path = "data_reconstruida/testing_data/"

#Lista imagenes reconstruidas
testing_data_reconstruida = []

# Cargar solo las primeras 500 imágenes válidas (reconstruidas)
max_images = 500
for i in range(1, 2587):
    if len(testing_data_reconstruida) >= max_images:
        break
    # Los archivos en la carpeta usan el prefijo 'testing_' en el nombre
    nombre_archivo = os.path.join(testing_data_reconstruida_path, f"testing_{i}.bmp")
    if not os.path.exists(nombre_archivo):
        print(f"Advertencia: no existe {nombre_archivo}")
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        testing_data_reconstruida.append(img)
        if len(testing_data_reconstruida) % 50 == 0 or len(testing_data_reconstruida) == max_images:
            print(f"Cargadas {len(testing_data_reconstruida)}/{max_images} imágenes reconstruidas")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes reconstruidas cargadas: {len(testing_data_reconstruida)}")

# Mostrar las primeras N imágenes cargadas en testing_data_reconstruida
N = 6
if isinstance(testing_data_reconstruida, list) and len(testing_data_reconstruida) > 0:
    n = min(N, len(testing_data_reconstruida))
    plt.figure(figsize=(3*n, 4))
    for i in range(n):
        img = testing_data_reconstruida[i]
        if img is None:
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax = plt.subplot(1, n, i+1)
        plt.imshow(disp, cmap=cmap)
        plt.title(f"#{i+1}  {disp.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("testing_data_reconstruida no es una lista o está vacía. Asegúrate de haber ejecutado la celda que carga las imágenes reconstruidas.")

"""Training data fold 0"""

# ...existing code...
# path (versión reconstruida)
training_data_fold_0_path_reconstruida = "data_reconstruida/training_data/fold_0/"

# Lista imágenes reconstruidas
training_data_fold_0_reconstruida = []

# Cargar máximo 500 imágenes reconstruidas
max_images = 500
for i in range(1, 10000):
    if len(training_data_fold_0_reconstruida) >= max_images:
        break
    nombre_archivo = os.path.join(training_data_fold_0_path_reconstruida, f"testing_fold_0_{i}.bmp")
    if not os.path.exists(nombre_archivo):
        # Si faltan muchos archivos secuenciales, simplemente continuar
        # (si prefieres detenerte al primer hueco, reemplaza continue por break)
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        training_data_fold_0_reconstruida.append(img)
        if len(training_data_fold_0_reconstruida) % 50 == 0 or len(training_data_fold_0_reconstruida) == max_images:
            print(f"Cargadas {len(training_data_fold_0_reconstruida)}/{max_images} imágenes reconstruidas (fold 0)")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes reconstruidas cargadas (fold 0): {len(training_data_fold_0_reconstruida)}")
# ...existing code...

# Mostrar las primeras N imágenes cargadas en testing_data_reconstruida
N = 6
if isinstance(training_data_fold_0_reconstruida, list) and len(training_data_fold_0_reconstruida) > 0:
    n = min(N, len(training_data_fold_0_reconstruida))
    plt.figure(figsize=(3*n, 4))
    for i in range(n):
        img = training_data_fold_0_reconstruida[i]
        if img is None:
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax = plt.subplot(1, n, i+1)
        plt.imshow(disp, cmap=cmap)
        plt.title(f"#{i+1}  {disp.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("training_data_fold_0_reconstruida no es una lista o está vacía. Asegúrate de haber ejecutado la celda que carga las imágenes reconstruidas.")

"""Training data fold 1"""

# path (versión reconstruida)
training_data_fold_1_path_reconstruida = "data_reconstruida/training_data/fold_1/"

# Lista imágenes reconstruidas
training_data_fold_1_reconstruida = []

# Cargar máximo 500 imágenes reconstruidas
max_images = 500
for i in range(1, 10000):
    if len(training_data_fold_1_reconstruida) >= max_images:
        break
    nombre_archivo = os.path.join(training_data_fold_1_path_reconstruida, f"testing_fold_1_{i}.bmp")
    if not os.path.exists(nombre_archivo):
        # Si falta el archivo, continuar buscando el siguiente índice
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        training_data_fold_1_reconstruida.append(img)
        if len(training_data_fold_1_reconstruida) % 50 == 0 or len(training_data_fold_1_reconstruida) == max_images:
            print(f"Cargadas {len(training_data_fold_1_reconstruida)}/{max_images} imágenes reconstruidas (fold 1)")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes reconstruidas cargadas (fold 1): {len(training_data_fold_1_reconstruida)}")

# Mostrar las primeras N imágenes cargadas en training_data_fold_1_reconstruida
N = 6
if isinstance(training_data_fold_1_reconstruida, list) and len(training_data_fold_1_reconstruida) > 0:
    n = min(N, len(training_data_fold_1_reconstruida))
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for idx in range(n):
        img = training_data_fold_1_reconstruida[idx]
        ax = axes[idx]
        if img is None:
            ax.set_title(f"#{idx+1} None")
            ax.axis('off')
            continue
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax.imshow(disp, cmap=cmap)
        ax.set_title(f"#{idx+1} {disp.shape}")
        ax.axis('off')
    for j in range(n, rows*cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("training_data_fold_1_reconstruida no es una lista o está vacía.")

"""Training data fold 2"""

# path (versión reconstruida)
training_data_fold_2_path_reconstruida = "data_reconstruida/training_data/fold_2/"

# Lista imágenes reconstruidas
training_data_fold_2_reconstruida = []

# Cargar máximo 500 imágenes reconstruidas
max_images = 500
for i in range(1, 10000):
    if len(training_data_fold_2_reconstruida) >= max_images:
        break
    nombre_archivo = os.path.join(training_data_fold_2_path_reconstruida, f"testing_fold_2_{i}.bmp")
    if not os.path.exists(nombre_archivo):
        # Si falta el archivo, continuar buscando el siguiente índice
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        training_data_fold_2_reconstruida.append(img)
        if len(training_data_fold_2_reconstruida) % 50 == 0 or len(training_data_fold_2_reconstruida) == max_images:
            print(f"Cargadas {len(training_data_fold_2_reconstruida)}/{max_images} imágenes reconstruidas (fold 2)")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes reconstruidas cargadas (fold 2): {len(training_data_fold_2_reconstruida)}")

# Mostrar las primeras N imágenes cargadas en training_data_fold_2_reconstruida
N = 6
if isinstance(training_data_fold_2_reconstruida, list) and len(training_data_fold_2_reconstruida) > 0:
    n = min(N, len(training_data_fold_2_reconstruida))
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for idx in range(n):
        img = training_data_fold_2_reconstruida[idx]
        ax = axes[idx]
        if img is None:
            ax.set_title(f"#{idx+1} None")
            ax.axis('off')
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax.imshow(disp, cmap=cmap)
        ax.set_title(f"#{idx+1} {disp.shape}")
        ax.axis('off')
    for j in range(n, rows*cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("training_data_fold_2_reconstruida no es una lista o está vacía.")

"""validation data"""

# Path (versión reconstruida)
validation_data_reconstruida_path = "data_reconstruida/validation_data/"

# Lista imágenes reconstruidas
validation_data_reconstruida = []

# Cargar máximo 500 imágenes reconstruidas
max_images = 500
for i in range(1, 1867):
    if len(validation_data_reconstruida) >= max_images:
        break
    nombre_archivo = os.path.join(validation_data_reconstruida_path, f"validation_{i}.bmp")
    if not os.path.exists(nombre_archivo):
        print(f"Advertencia: no existe {nombre_archivo}")
        continue
    img = cv2.imread(nombre_archivo)
    if img is not None:
        validation_data_reconstruida.append(img)
        if len(validation_data_reconstruida) % 50 == 0 or len(validation_data_reconstruida) == max_images:
            print(f"Cargadas {len(validation_data_reconstruida)}/{max_images} imágenes reconstruidas (validation)")
    else:
        print(f"Advertencia: no se pudo cargar la imagen {nombre_archivo}")

print(f"Total imágenes reconstruidas cargadas (validation): {len(validation_data_reconstruida)}")

# Mostrar las primeras N imágenes cargadas en validation_data_reconstruida
N = 6
if isinstance(validation_data_reconstruida, list) and len(validation_data_reconstruida) > 0:
    n = min(N, len(validation_data_reconstruida))
    plt.figure(figsize=(3*n, 4))
    for i in range(n):
        img = validation_data_reconstruida[i]
        if img is None:
            continue
        # Si es BGR (cv2) convertir a RGB para matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            disp = img
            cmap = 'gray'
        ax = plt.subplot(1, n, i+1)
        plt.imshow(disp, cmap=cmap)
        plt.title(f"#{i+1}  {disp.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("validation_data_reconstruida no es una lista o está vacía.")

