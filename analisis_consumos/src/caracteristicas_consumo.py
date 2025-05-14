"""
Descripción:
Obtener caracteristicas de consumos de cada CUPS en funcion de su historico: medias, maximos, ...

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python caracteristicas_consumos.py, (csv con datos de hasta dos años de consumo por horas)
    Devuelve las características de consumo de ese csv
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""
import pandas as pd
from tfm.utils.data_frame import add_timestamp_values

def caracteristicas_consumo(csv):
    datos_consumo = pd.read_csv(csv, sep=";")
    
       
    #print(datos_consumo.columns)
    datos_consumo.head()

    # 2. Arreglar posibles 24:00 en 'time'
    if 'time' in datos_consumo.columns:
        datos_consumo['time'] = datos_consumo['time'].replace('24:00', '00:00')

    # 3. Crear la columna 'timestamp'
    datos_consumo['timestamp'] = pd.to_datetime(datos_consumo['date'] + ' ' + datos_consumo['time'], 
                                                format='%d/%m/%Y %H:%M', errors='coerce')

    # 4. Añadir columnas de hora, día de la semana, mes
    datos_consumo = add_timestamp_values(datos_consumo, 'timestamp')

    # 5. Mostrar para comprobar
    #print(datos_consumo.head())
    

    """#Crear timestamp

    datos_consumo['time'] = datos_consumo['time'].replace('24:00','00:00')
    datos_consumo['timestamp'] = pd.to_datetime(datos_consumo['date'] + ' ' + datos_consumo['time'])
    datos_consumo.head()"""

    """#Caracteristicas básicas

    Consumo medio

    Desviación estándar del consumo

    Consumo máximo y mínimo

    Consumo en percentiles

    Consumo total acumulado
    """

    media_consumo = datos_consumo['consumptionKWh'].mean()
    media_consumo

    std_consumo = datos_consumo['consumptionKWh'].std()
    std_consumo

    max_consumo = datos_consumo['consumptionKWh'].max()
    min_consumo = datos_consumo['consumptionKWh'].min()
    min_consumo, max_consumo

    percentil_25_consumo = datos_consumo['consumptionKWh'].quantile(0.25)
    percentil_50_consumo = datos_consumo['consumptionKWh'].quantile(0.5)
    percentil_75_consumo = datos_consumo['consumptionKWh'].quantile(0.75)

    percentil_25_consumo, percentil_50_consumo, percentil_75_consumo

    sum_consumo = datos_consumo['consumptionKWh'].sum()
    sum_consumo

    """#Características temporales

    Horarios

    Semanales

    Estacionales

    ##Promedio por hora
    """

    consumo_por_horas = datos_consumo.groupby('hour')['consumptionKWh'].mean()
    # Crear el diccionario formateado
    media_consumos_horas = {
        f'consumo_{str(hora).zfill(2)}': valor for hora, valor in consumo_por_horas.items()
    }

    df_media_consumos = pd.DataFrame([media_consumos_horas])

    """##Por ventanas de tiempo del dia

    (Mañana, tarde, noche, madrugada)
    """

    ventanas_tiempo = {
        'Mañana': range(6, 12),
        'Tarde': range(12, 18),
        'Noche': range(18, 24),
        'Madrugada': range(0, 6)
    }

    medias_por_periodo = {
        periodo: consumo_por_horas[consumo_por_horas.index.isin(horas)].mean()
        for periodo, horas in ventanas_tiempo.items()
    }

    df_medias_por_periodo = pd.DataFrame([medias_por_periodo])
    

    """##Por día de la semana"""

    media_por_dia = datos_consumo.groupby('dayofweek')['consumptionKWh'].mean()
    dias_nombre = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    media_por_dia = media_por_dia.rename(index=dias_nombre)
    df_media_por_dia = media_por_dia.to_frame().T

    

    """## Entresemana o finde"""
    datos_consumo['grupo_dia'] = datos_consumo['dayofweek'].apply(lambda x: 'Fin de semana' if x >= 5 else 'Entre semana')
    media_por_grupo = datos_consumo.groupby('grupo_dia')['consumptionKWh'].mean()
    df_media_por_grupo = media_por_grupo.to_frame().T  # Asegura que "Entre semana" y "Fin de semana" sean columnas

    """## Por estación"""
    datos_consumo['estacion'] = datos_consumo['month'].map(
        lambda x: 'invierno' if x in [12, 1, 2] else 
                'primavera' if x in [3, 4, 5] else 
                'verano' if x in [6, 7, 8] else 'otoño'
    )

    """##Por estación"""
    estaciones = datos_consumo.groupby('estacion')['consumptionKWh'].mean()
    df_estaciones = estaciones.to_frame().T

   

    """#Promedio de consumo total en un día"""

    promedio_por_dia = datos_consumo.groupby(datos_consumo['timestamp'].dt.date)['consumptionKWh'].mean().mean()

    promedio_por_dia

    """#Sumatorios
    Por mes distinguiendo años

    Por semana
    """

    # Consumo mensual
    consumo_mensual = datos_consumo.groupby(datos_consumo['timestamp'].dt.to_period('M'))['consumptionKWh'].sum()
    df_consumo_mensual = consumo_mensual.to_frame().T  # Convierte los meses en columnas

    # Consumo semanal
    consumo_semanal = datos_consumo.groupby(datos_consumo['timestamp'].dt.to_period('W'))['consumptionKWh'].sum()
    df_consumo_semanal = consumo_semanal.to_frame().T  # Convierte las semanas en columnas


    # Organizar los resultados en un diccionario
    resultado = {
        'media_consumo': media_consumo,
        'std_consumo':std_consumo,
        'min_consumo':min_consumo,
        'max_consumo':max_consumo,
        'percentil_25_consumo':percentil_25_consumo,
        'percentil_50_consumo':percentil_50_consumo,
        'percentil_75_consumo':percentil_75_consumo,
        'sum_consumo':sum_consumo,
        
        'promedio_por_dia': promedio_por_dia    
        
    }
    
    df_resultado = pd.DataFrame([resultado])
    for df in [df_media_consumos, df_medias_por_periodo, df_media_por_dia, 
           df_media_por_grupo, df_estaciones, df_consumo_mensual, 
           df_consumo_semanal, df_resultado]:
        df.index = [0] 
    resultados = pd.concat([df_resultado, df_media_consumos, df_medias_por_periodo,
                            df_media_por_dia, df_media_por_grupo,df_estaciones,
                            df_consumo_mensual, df_consumo_semanal], axis=1)
    print(type(resultados))
    #resultados.to_csv("resultados_finales.csv", index=True)
    


    return resultados


    