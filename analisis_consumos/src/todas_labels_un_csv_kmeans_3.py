#!/usr/bin/env python3
import os

def merge_clusters(root_dir, output_path):
    """
    Lee todos los CSV de root_dir (que tienen dos columnas: 'hogar' y 'cluster')
    y genera un único CSV en output_path con:
    - Primera columna: 'hogar'
    - Columnas siguientes: cada ventana (nombre del fichero, sin .csv)
    - Cada celda: el cluster al que pertenece el hogar en esa ventana
    """
    clusters_per_hogar = {}   # { hogar: { ventana: cluster, … } }
    ventanas = []

    for fname in sorted(os.listdir(root_dir)):
        if not fname.lower().endswith('.csv'):
            continue

        ventana = os.path.splitext(fname)[0]
        ventanas.append(ventana)
        path = os.path.join(root_dir, fname)

        # Leemos líneas (primero intentando utf-8-sig, luego latin-1 si hay error)
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = f.read().splitlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                lines = f.read().splitlines()
        if not lines:
            continue

        # Detectamos delimitador: más ';' o más ',' en la cabecera
        header = lines[0]
        delim = ';' if header.count(';') > header.count(',') else ','
        cols = [h.strip().lower().lstrip('\ufeff') for h in header.split(delim)]
        if 'hogar' not in cols or 'cluster' not in cols:
            print(f"Ignorando «{fname}»: no encontró columnas 'hogar' y 'cluster'")
            continue

        idx_hogar = cols.index('hogar')
        idx_cluster = cols.index('cluster')

        # Procesamos cada fila de datos
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(delim)]
            if len(parts) <= max(idx_hogar, idx_cluster):
                continue
            hogar  = parts[idx_hogar]
            cluster = parts[idx_cluster]
            if not hogar:
                continue
            clusters_per_hogar.setdefault(hogar, {})[ventana] = cluster

    # Escribimos el CSV combinado (solo un encabezado de 'hogar' + ventanas)
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write('hogar,' + ','.join(ventanas) + '\n')
        for hogar in sorted(clusters_per_hogar):
            fila = [hogar] + [clusters_per_hogar[hogar].get(v, '') for v in ventanas]
            out.write(','.join(fila) + '\n')

    print(f"CSV combinado guardado en: {output_path}")


if __name__ == '__main__':
    # ————— CONFIGURACIÓN —————
    dir_csv    = 'resultados/imagenes_redaccion/labels_hogares'
    salida_csv = 'resultados/imagenes_redaccion/resultado_labels_ventanas.csv'
    # ————————————————————————

    merge_clusters(dir_csv, salida_csv)
