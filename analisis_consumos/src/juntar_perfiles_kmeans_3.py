#!/usr/bin/env python3
import os

def juntar_perfiles(root_dir, output_file):
    """
    Lee todos los CSV de root_dir con el formato:
        (encabezado) variable,cluster_0,cluster_1,…,cluster_N
        (datos…)   media_consumo,0.17,0.29,…,0.42
                   std_consumo,0.19,0.33,…,0.48
                   …
    y genera un CSV único en output_file que queda así:
        ventana,variable,cluster_0,cluster_1,…,cluster_N
        ventanaA,media_consumo,0.17,0.29,…,0.42
        ventanaA,std_consumo,0.19,0.33,…,0.48
        …
        ventanaB,media_consumo,0.21,0.31,…,0.47
        …
    donde “ventana” viene del nombre de fichero (sin “.csv”).
    """
    primeros = True
    with open(output_file, 'w', encoding='utf-8') as fout:
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith('.csv'):
                continue

            ventana = os.path.splitext(fname)[0]
            path = os.path.join(root_dir, fname)
            # Leemos eliminando posible BOM
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = f.read().splitlines()
            if not lines:
                continue

            # Detectamos delimitador: más ';' o más ',' en la cabecera
            header = lines[0]
            delim = ';' if header.count(';') > header.count(',') else ','

            # Procesamos el encabezado
            cols = [h.strip() for h in header.split(delim)]
            # En la primera iteración escribimos el encabezado completo
            if primeros:
                fout.write('ventana,' + ','.join(cols) + '\n')
                primeros = False

            # Recorremos cada línea de datos (saltando la cabecera)
            for row in lines[1:]:
                if not row.strip():
                    continue
                valores = [v.strip() for v in row.split(delim)]
                # Escribimos: ventana + valores de la fila
                fout.write(ventana + ',' + ','.join(valores) + '\n')

    print(f"CSV combinado de perfiles guardado en: {output_file}")


if __name__ == '__main__':
    # ————— CONFIGURA AQUÍ —————
    directorio_csv = 'resultados/imagenes_redaccion/profile'         # carpeta con los CSV originales
    csv_salida     = 'resultados/imagenes_redaccion/perfiles_combinados.csv'  # fichero resultante
    # ————————————————————————

    juntar_perfiles(directorio_csv, csv_salida)

