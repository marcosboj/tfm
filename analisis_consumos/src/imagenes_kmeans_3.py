#!/usr/bin/env python3
import os

def copiar_imagenes(root, destino, nombre_imagen, prefijo_a_quitar=""):
    # Asegurarnos de que exista la carpeta destino
    if not os.path.exists(destino):
        os.makedirs(destino)
    
    for nombre in os.listdir(root):
        ruta_carpeta = os.path.join(root, nombre)
        if not os.path.isdir(ruta_carpeta):
            continue
        
        # Si el nombre comienza por el prefijo, lo recortamos
        if prefijo_a_quitar and nombre.startswith(prefijo_a_quitar):
            # cortamos el prefijo y eliminamos guiones/barras bajos sobrantes
            nombre_limpio = nombre[len(prefijo_a_quitar):].lstrip("_- ")
        else:
            nombre_limpio = nombre
        
        origen = os.path.join(ruta_carpeta, nombre_imagen)
        if not os.path.isfile(origen):
            print(f"No existe {nombre_imagen} en {ruta_carpeta}")
            continue
        
        # construimos el nombre destino con el nombre limpio
        nombre_destino = f"{nombre_limpio}_{nombre_imagen}"
        ruta_destino = os.path.join(destino, nombre_destino)
        
        # copiamos en modo binario
        with open(origen, 'rb') as f_src, open(ruta_destino, 'wb') as f_dst:
            while buf := f_src.read(8192):
                f_dst.write(buf)
        
        print(f"Copiado: {origen} → {ruta_destino}")
if __name__ == "__main__":
    # OPCIÓN A: defines las rutas aquí:
    dir_origen = "resultados/clustering_resumen"
    dir_destino = "resultados/imagenes_redaccion"
    img_name = "KMeans_k3_profile.csv"
    prefijo_comun     = "resumen_consumos"

    copiar_imagenes(dir_origen, dir_destino, img_name, prefijo_comun)
