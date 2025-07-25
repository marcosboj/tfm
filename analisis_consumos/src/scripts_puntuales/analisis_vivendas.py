"""
Descripción:
Análisis de datos de la encuesta realizada a los usuarios de los consumos. Busqueda de correlaciones.

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python analisis_viviendas.py,encuesta_consumos_oliver.csv
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


path_actual = Path.cwd()
print(path_actual)
# Construir el path del archivo en la carpeta anterior (por ejemplo, archivo.csv)
archivo = path_actual / "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/encuesta_consumos_oliver.csv"
print(archivo)

datos_categoricos_viviendas = pd.read_csv(archivo)

datos_categoricos_viviendas.shape

datos_categoricos_viviendas.columns

"""# Analisis por variables en relación a la gente que vive en las casas

Columnas en relacion a cuanta gente vive y de que tipo:
 ¿Adultos en la vivienda? [Ocupados/trabajando] texto en negrita

*   ¿Adultos en la vivienda? [Ocupados/trabajando]
*   ¿Adultos en la vivienda? [Realizando tareas del hogar]
*   ¿Adultos en la vivienda? [Parados/as]
*   ¿Adultos en la vivienda? [Jubilados/as - pensionistas de algún tipo]
*   ¿Adultos en la vivienda? [Estudiantes]
*   Menores en la vivienda [Menores en la vivienda]
"""

datos_categoricos_viviendas["¿Adultos en la vivienda? [Ocupados/trabajando]"].value_counts()

datos_categoricos_viviendas["¿Adultos en la vivienda? [Ocupados/trabajando]"].value_counts().sort_index().plot(kind='bar')

"""Hacerlo con todas a la vez"""

columnas_num_personas = ["¿Adultos en la vivienda? [Ocupados/trabajando]","¿Adultos en la vivienda? [Realizando tareas del hogar]", "¿Adultos en la vivienda? [Parados/as]",
            "¿Adultos en la vivienda? [Jubilados/as - pensionistas de algún tipo]", "¿Adultos en la vivienda? [Estudiantes]", "Menores en la vivienda [Menores en la vivienda]"]

fig, ax = plt.subplots(nrows=len(columnas_num_personas), figsize=(6, 5 * len(columnas_num_personas)))
# Iterar sobre las columnas y graficar
for i, col in enumerate(columnas_num_personas):
    datos_categoricos_viviendas[col].value_counts().sort_index().plot(kind="bar", ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xlabel("Valor")
    ax[i].set_ylabel("Frecuencia")
    ax[i].tick_params(axis="x", rotation=45)

# Ajustar diseño
plt.tight_layout()
plt.show()

for col in columnas_num_personas:
  print(datos_categoricos_viviendas[col].value_counts().sort_index())

"""# Análisis percepción de consumo

## Entresemana
'Percepción de distribución de consumos [Entre semana]',
"""

todas_percepcion_entre = datos_categoricos_viviendas['Percepción de distribución de consumos [Entre semana]'].str.split(',')
#print(todas_percepcion_entre)
todas_percepcion_entre_flat = todas_percepcion_entre.explode().str.strip()
#print(todas_percepcion_entre_flat)
resumen_respuestas_entre = todas_percepcion_entre_flat.value_counts()
print(resumen_respuestas_entre)

"""##Finde
'Percepción de distribución de consumos [Fin de semana]',

"""

todas_percepcion_finde = datos_categoricos_viviendas['Percepción de distribución de consumos [Fin de semana]'].str.split(',')
#print(todas_percepcion_entre)
todas_percepcion_finde_flat = todas_percepcion_finde.explode().str.strip()
#print(todas_percepcion_entre_flat)
resumen_respuestas_finde = todas_percepcion_finde_flat.value_counts()
print(resumen_respuestas_finde)

"""# Metros cuadrados (m2)

Metros cuadrados aproximados de la vivienda
"""

import seaborn as sns
plt.figure(figsize=(10, 6))

# Crear el histograma con seaborn
sns.histplot(datos_categoricos_viviendas['Metros cuadrados aproximados de la vivienda'], bins=10, color='skyblue')

# Añadir título y etiquetas
plt.title('Histograma de Metros Cuadrados')
plt.xlabel('Metros Cuadrados')
plt.ylabel('Frecuencia')

# Mostrar el gráfico
plt.show()



"""# Suministros eléctricos

'¿Tienes calefacción eléctrica?',
"""

todas_calef = datos_categoricos_viviendas['¿Tienes calefacción eléctrica?'].str.split(',')
todas_calef_flat = todas_calef.explode().str.strip()
resumen_respuestas_calef = todas_calef_flat.value_counts()
print(resumen_respuestas_calef)

"""'¿Tienes sistema de agua caliente eléctrica?'"""

todas_acs = datos_categoricos_viviendas['¿Tienes sistema de agua caliente eléctrica?'].str.split(',')
todas_acs_flat = todas_acs.explode().str.strip()
resumen_respuestas_acs = todas_acs_flat.value_counts()
print(resumen_respuestas_acs)

"""# Por equipo

Columnas en relacion a equipos:
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Cocina eléctrica (no de gas/but,etc)]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Horno eléctrico]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Microondas]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Lavavajillas]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Lavadora]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Secadora]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Frigorífico con congelador]',
- 'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Aire acondicionado]',
"""

columnas_equipos = ['Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Cocina eléctrica (no de gas/but,etc)]',
                    'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Horno eléctrico]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Microondas]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Lavavajillas]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Lavadora]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Secadora]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Frigorífico con congelador]',
'Indica si en tu vivienda dispones de los siguientes electrodomésticos o equipos: [Aire acondicionado]',]

for col in columnas_equipos:
  print(datos_categoricos_viviendas[col].value_counts().sort_index())



"""# Correlaciones entre variables

Ajustar las columnas multi opcion antes de nada
"""

datos_categoricos_viviendas_copy = datos_categoricos_viviendas.copy()
# Eliminar las columnas que no quieres que correlacionen
columnas_a_eliminar = ['Marca temporal', 'DNI/NIE/NIF (del titular)','¿Crees que podrías cambiar tus hábitos de consumo para ajustarlos a horas de más producción solar?'
, 'Teléfono','Tienes la función de programar alguno de tus electrodomésticos',
'Observaciones (apuntar cosas que nombren en la conversación pero que no están en ninguna de las preguntas y pueda resultar útil ya sea sobre el consumo o sobre los hábitos que tienen)']
datos_sin_columnas = datos_categoricos_viviendas_copy.drop(columns=columnas_a_eliminar)


columnas_multi = [
       'Percepción de distribución de consumos [Entre semana]',
       'Percepción de distribución de consumos [Fin de semana]',
       '¿Tienes calefacción eléctrica?',
       '¿Tienes sistema de agua caliente eléctrica?',
       ]

for col in columnas_multi: #recorre la lista que hemos creado con las columnas que son multi
  opciones_unicas = set() #set elimina duplicados, en este momento se crea una lista vacia. Mas adelante tendra cada opcion de respuesta por cada columna
  for respuestas in datos_sin_columnas[col].dropna():#recorre la serie de la col que en la que estamos
    opciones_unicas.update(respuestas.split(', '))#en la celda en la que esta(respuesta) divide por ' si tiene y hace una lista de dos objetos semete en la lista opciones unicas la respuesta en la que estamos
  for opcion in opciones_unicas: #recorre la listas de opciones unicas
    datos_sin_columnas[f'{col}_{opcion}'] = datos_sin_columnas[col].apply(lambda x: 1 if opcion in str(x) else 0)
    #se crea una nueva columna con el nombre col_opcion
    #datos_sin_columnas[col] Obtiene la columna original con respuestas múltiples
    # lambda Si opcion está en x, devuelve 1 (significa que la opción está presente en la celda).

datos_sin_columnas = datos_sin_columnas.drop(columns=columnas_multi)

datos_sin_columnas = datos_sin_columnas.replace({'Si': 1, 'No': 0}).astype(int, errors='ignore')

datos_sin_columnas

correlaciones = datos_sin_columnas.corr()

# Mostrar la matriz de correlación
print(correlaciones)


import seaborn as sns
import matplotlib.pyplot as plt

# Crear la figura y el mapa de calor para la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Mostrar el gráfico
plt.title('Mapa de Calor de Correlaciones entre Variables')
plt.show()

