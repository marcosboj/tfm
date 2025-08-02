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
#caracteristicas_consumo.py
import pandas as pd
import numpy as np

from tfm.utils.data_frame import add_timestamp_values

def caracteristicas_consumo(csv, filtro="completo"):
    # 1. Leer CSV
    datos_consumo = pd.read_csv(csv, sep=";")
    
    # —————————————— Creación avanzada de timestamp ——————————————
    # 1) Series de fechas datetime (para el cálculo interno, NO creamos columna aún)
    _dates = pd.to_datetime(datos_consumo['date'], dayfirst=True)
    
    # 2) Sustituimos “24:00” o “24:00:00” por “00:00” solo para parsear
    _times = datos_consumo['time'].replace({'24:00:00': '00:00', '24:00': '00:00'})
    
    # 3) Construimos el timestamp local, sumando 1 día si time era “24:00”
    _full_local = (
        pd.to_datetime(
            _dates.dt.strftime('%Y-%m-%d') + ' ' + _times,
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )
        + pd.to_timedelta(datos_consumo['time'].isin(['24:00','24:00:00']).astype(int), unit='d')
    )
    
    # 4) Localizamos en Europe/Madrid y convertimos a UTC, guardando en la misma columna
    datos_consumo['timestamp'] = (
        _full_local
        .dt.tz_localize('Europe/Madrid', ambiguous=False, nonexistent='shift_forward')
        .dt.tz_convert('UTC')
    )
    # ————————————————————————————————————————————————————————————————
    # justo después de tener datos_consumo['timestamp']  
    festivos = pd.read_csv('data/festivos_zgz.csv')['fecha'].astype(str).tolist()
    datos_consumo['date_only'] = datos_consumo['timestamp'].dt.strftime('%Y-%m-%d')
    datos_consumo['weekday']   = datos_consumo['timestamp'].dt.weekday
    datos_consumo['day_type']  = np.where(
        datos_consumo['date_only'].isin(festivos) | datos_consumo['weekday'].isin([5,6]),
        'festivo','laborable'
    )

     # 5. Filtrar por rango de fechas deseado (UTC)
    start_date = pd.Timestamp("2024-07-01 00:00", tz="UTC")
    end_date = pd.Timestamp("2025-06-30 23:00", tz="UTC")
    datos_consumo = datos_consumo[(datos_consumo['timestamp'] >= start_date) & (datos_consumo['timestamp'] <= end_date)]
    print(f"Filtrado por fecha: quedan {len(datos_consumo)} filas entre {start_date.date()} y {end_date.date()}.")

    # 5. Añadir columnas de hora, día de la semana, mes, etc.
    datos_consumo = add_timestamp_values(datos_consumo, 'timestamp')


    # Aplicar filtro según tipo
    match filtro:
        case "completo":
            df_filtrado = datos_consumo.copy()
            nombre_filtro = "todo"
        case "fechas":
            fecha_inicio = "2024-07-01"
            fecha_fin = "2025-06-30"
            df_filtrado = datos_consumo[
                (datos_consumo["timestamp"] >= fecha_inicio) &
                (datos_consumo["timestamp"] < fecha_fin)
            ]
            nombre_filtro = f"{fecha_inicio}_a_{fecha_fin}"
        case "mes":
            mes, año = 6, 2025
            df_filtrado = datos_consumo[
                (datos_consumo["timestamp"].dt.month == mes) &
                (datos_consumo["timestamp"].dt.year == año)
            ]
            nombre_filtro = f"mes_{mes:02d}_{año}"
        case "meses":
            mes = 12  # por ejemplo junio; podrías recibirlo como parámetro
            # filtramos **solo** por mes, sin year
            df_filtrado = datos_consumo[
                datos_consumo["timestamp"].dt.month == mes
            ]
            nombre_filtro = f"meses_{mes:02d}"           
        case "estacion":
            estacion = "otoño"
            datos_consumo["estacion"] = datos_consumo["timestamp"].dt.month.map(
                lambda x: "invierno" if x in [12, 1, 2] else
                          "primavera" if x in [3, 4, 5] else
                          "verano" if x in [6, 7, 8] else
                          "otoño"
            )
            df_filtrado = datos_consumo[
                datos_consumo["estacion"] == estacion
            ]
            nombre_filtro = f"estacion_{estacion}"
        case "dia_semana":
            # 0=lunes, 1=martes, …, 6=domingo
            dia = 6  # cámbialo por el número de día que quieras
            nombres = ['lunes','martes','miercoles','jueves','viernes','sabado','domingo']
            df_filtrado = datos_consumo[
                datos_consumo["timestamp"].dt.weekday == dia
            ]
            nombre_filtro = f"dia_{nombres[dia]}"
        case "tipo_dia":
            tipo = "laborable" #(laborable festivo)
            # filtra por day_type, usando la columna que ya creaste (laborable festivo)
            df_filtrado   = datos_consumo[
                datos_consumo["day_type"] == tipo
            ]
            nombre_filtro = f"tipo_dia_{tipo}"
        case _:
            print("Filtro no reconocido")
            df_filtrado = datos_consumo.copy()
            nombre_filtro = "desconocido"

    # Mapear estación sin función
    df_filtrado["estacion"] = df_filtrado["timestamp"].dt.month.map(
        lambda x: "invierno" if x in [12, 1, 2] else
                "primavera" if x in [3, 4, 5] else
                "verano" if x in [6, 7, 8] else
                "otoño"
    )

    """#Caracteristicas básicas

    Consumo medio

    Desviación estándar del consumo

    Consumo máximo y mínimo

    Consumo en percentiles

    Consumo total acumulado
    """

    media_consumo = df_filtrado['consumptionKWh'].mean()
    media_consumo

    std_consumo = df_filtrado['consumptionKWh'].std()
    std_consumo

    max_consumo = df_filtrado['consumptionKWh'].max()
    min_consumo = df_filtrado['consumptionKWh'].min()
    min_consumo, max_consumo

    percentil_25_consumo = df_filtrado['consumptionKWh'].quantile(0.25)
    percentil_50_consumo = df_filtrado['consumptionKWh'].quantile(0.5)
    percentil_75_consumo = df_filtrado['consumptionKWh'].quantile(0.75)

    percentil_25_consumo, percentil_50_consumo, percentil_75_consumo

    sum_consumo = df_filtrado['consumptionKWh'].sum()
    sum_consumo

    """#Características temporales

    Horarios

    Semanales

    Estacionales

    ##Promedio por hora
    """

    consumo_por_horas = df_filtrado.groupby('hour')['consumptionKWh'].mean()
    # Crear el diccionario formateado
    media_consumos_horas = {
        f'consumo_{str(hora).zfill(2)}': valor for hora, valor in consumo_por_horas.items()
    }

    df_media_consumos = pd.DataFrame([media_consumos_horas])

    # Agrupar por hora y calcular la suma
    suma_por_hora = df_filtrado.groupby('hour')['consumptionKWh'].sum()
    suma_consumos_horas = {
        f'consumo_{str(hora).zfill(2)}_suma': valor for hora, valor in suma_por_hora.items()
    }
    df_suma_consumos = pd.DataFrame([suma_consumos_horas])

    std_por_hora = df_filtrado.groupby('hour')['consumptionKWh'].std()
    suma_consumos_horas = {
        f'consumo_{str(hora).zfill(2)}_std': valor for hora, valor in suma_por_hora.items()
    }
    df_std_consumos = pd.DataFrame([suma_consumos_horas])


    """##Por ventanas de tiempo del dia

    (Mañana, tarde, noche, madrugada)
    """

    ventanas_tiempo = {
        'Mañana': range(6, 10),
        'Mediodia': range(10, 16),
        'Tarde': range(16, 20),
        'Noche': range(20, 24),
        'Madrugada': range(0, 6)
    }

    medias_por_periodo = {
        periodo: consumo_por_horas[consumo_por_horas.index.isin(horas)].mean()
        for periodo, horas in ventanas_tiempo.items()
    }
    df_medias_por_periodo = pd.DataFrame([medias_por_periodo])

    s_ventanas_tiempo = {
        's_Mañana': range(6, 10),
        's_Mediodia': range(10, 16),
        's_Tarde': range(16, 20),
        's_Noche': range(20, 24),
        's_Madrugada': range(0, 6)
    }
    sumas_por_periodo = {
    periodo: suma_por_hora[suma_por_hora.index.isin(horas)].sum()
        for periodo, horas in s_ventanas_tiempo.items()
    }

    df_sumas_por_periodo = pd.DataFrame([sumas_por_periodo])

    std_ventanas_tiempo = {
        'std_Mañana': range(6, 10),
        'std_Mediodia': range(10, 16),
        'std_Tarde': range(16, 20),
        'std_Noche': range(20, 24),
        'std_Madrugada': range(0, 6)
    }
    std_por_periodo = {
    periodo: std_por_hora[std_por_hora.index.isin(horas)].std()
        for periodo, horas in std_ventanas_tiempo.items()
    }

    df_std_por_periodo = pd.DataFrame([std_por_periodo])
    

    """##Por día de la semana"""
    
    media_por_dia = df_filtrado.groupby('dayofweek')['consumptionKWh'].mean()
    dias_nombre = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    media_por_dia = media_por_dia.rename(index=dias_nombre)
    df_media_por_dia = media_por_dia.to_frame().T
    
    # Agrupar por día de la semana y sumar el consumo
    suma_por_dia = df_filtrado.groupby('dayofweek')['consumptionKWh'].sum()
    s_dias_nombre = {0: 's_Lunes', 1: 's_Martes', 2: 's_Miércoles', 3: 's_Jueves', 4: 's_Viernes', 5: 's_Sábado', 6: 's_Domingo'}
    suma_por_dia = suma_por_dia.rename(index=s_dias_nombre)
    # Convertir a DataFrame en una sola fila
    df_suma_por_dia = suma_por_dia.to_frame().T

    std_por_dia = df_filtrado.groupby('dayofweek')['consumptionKWh'].std()
    std_dias_nombre = {0: 'std_Lunes', 1: 'std_Martes', 2: 'std_Miércoles', 3: 'std_Jueves', 4: 'std_Viernes', 5: 'std_Sábado', 6: 'std_Domingo'}
    std_por_dia = std_por_dia.rename(index=std_dias_nombre)
    df_std_por_dia = std_por_dia.to_frame().T

    

    """## Entresemana o finde"""
    df_filtrado['grupo_dia'] = df_filtrado['dayofweek'].apply(lambda x: 'Fin de semana' if x >= 5 else 'Entre semana')
    media_por_grupo = df_filtrado.groupby('grupo_dia')['consumptionKWh'].mean()
    df_media_por_grupo = media_por_grupo.to_frame().T  # Asegura que "Entre semana" y "Fin de semana" sean columnas
    df_filtrado['s_grupo_dia'] = df_filtrado['dayofweek'].apply(lambda x: 's_Fin de semana' if x >= 5 else 's_Entre semana')
    suma_por_grupo = df_filtrado.groupby('s_grupo_dia')['consumptionKWh'].sum()
    df_suma_por_grupo = suma_por_grupo.to_frame().T

    df_filtrado['std_grupo_dia'] = df_filtrado['dayofweek'].apply(lambda x: 'std_Fin de semana' if x >= 5 else 'std_Entre semana')
    std_por_grupo = df_filtrado.groupby('std_grupo_dia')['consumptionKWh'].std()
    df_std_por_grupo = std_por_grupo.to_frame().T


    """## Por estación"""
    df_filtrado['estacion'] = df_filtrado['month'].map(
        lambda x: 'invierno' if x in [12, 1, 2] else 
                'primavera' if x in [3, 4, 5] else 
                'verano' if x in [6, 7, 8] else 'otoño'
    )

    """##Por estación"""
    estaciones = df_filtrado.groupby('estacion')['consumptionKWh'].mean()
    df_estaciones = estaciones.to_frame().T
    df_filtrado['s_estacion'] = df_filtrado['month'].map(
        lambda x: 's_invierno' if x in [12, 1, 2] else 
                's_primavera' if x in [3, 4, 5] else 
                's_verano' if x in [6, 7, 8] else 's_otoño'
    )

    suma_estaciones = df_filtrado.groupby('s_estacion')['consumptionKWh'].sum()
    df_suma_estaciones = suma_estaciones.to_frame().T


    df_filtrado['std_estacion'] = df_filtrado['month'].map(
        lambda x: 'std_invierno' if x in [12, 1, 2] else 
                'std_primavera' if x in [3, 4, 5] else 
                'std_verano' if x in [6, 7, 8] else 'std_otoño'
    )

    std_estaciones = df_filtrado.groupby('std_estacion')['consumptionKWh'].std()
    df_std_estaciones = std_estaciones.to_frame().T

   

    """#Promedio de consumo total en un día"""

    promedio_por_dia = df_filtrado.groupby(df_filtrado['timestamp'].dt.date)['consumptionKWh'].mean().mean()

    promedio_por_dia

    # Agrupar por fecha (ignorando la hora) y sumar el consumo de cada día
    consumo_total_por_dia = df_filtrado.groupby(df_filtrado['timestamp'].dt.date)['consumptionKWh'].sum()

    # Calcular el promedio de esos totales diarios
    consumo_medio_diario = consumo_total_por_dia.mean()

    """#Sumatorios
    Por mes distinguiendo años

    Por semana
    """

    # Consumo mensual
    consumo_mensual = df_filtrado.groupby(df_filtrado['timestamp'].dt.to_period('M'))['consumptionKWh'].sum()
    df_consumo_mensual = consumo_mensual.to_frame().T  # Convierte los meses en columnas

    consumo_por_mes = df_filtrado.groupby(df_filtrado['timestamp'].dt.month)['consumptionKWh'].sum()
    df_consumo_por_mes = consumo_por_mes.to_frame().T
    meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
         7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
    df_consumo_por_mes.rename(columns=meses, inplace=True)

    # Consumo semanal
    consumo_semanal = df_filtrado.groupby(df_filtrado['timestamp'].dt.to_period('W'))['consumptionKWh'].sum()
    df_consumo_semanal = consumo_semanal.to_frame().T  # Convierte las semanas en columnas
    


 
    resultado = {
        'media_consumo': media_consumo,
        'std_consumo':std_consumo,
        'min_consumo':min_consumo,
        'max_consumo':max_consumo,
        'percentil_25_consumo':percentil_25_consumo,
        'percentil_50_consumo':percentil_50_consumo,
        'percentil_75_consumo':percentil_75_consumo,
        'sum_consumo':sum_consumo,
        
        'promedio_por_dia': promedio_por_dia,
        'consumo_medio_diario' : consumo_medio_diario    
    }
    
    df_resultado = pd.DataFrame([resultado])
    for df in [df_media_consumos,df_suma_consumos,df_std_consumos, df_medias_por_periodo,df_sumas_por_periodo,
                df_std_por_periodo,df_media_por_dia,df_suma_por_dia,df_std_por_dia, df_media_por_grupo,df_suma_por_grupo,
                df_std_por_grupo, df_estaciones,
                df_suma_estaciones,df_std_estaciones, df_consumo_mensual,df_consumo_por_mes, df_consumo_semanal, df_resultado]:
        df.index = [0] 
    resultados = pd.concat([df_resultado, df_media_consumos,df_suma_consumos,df_std_consumos, df_medias_por_periodo,
                            df_sumas_por_periodo,df_std_por_periodo,df_media_por_dia,df_suma_por_dia,df_std_por_dia, df_media_por_grupo,
                            df_suma_por_grupo, df_std_por_grupo, df_estaciones,df_suma_estaciones,df_std_estaciones, df_consumo_mensual,
                            df_consumo_por_mes, df_consumo_semanal], axis=1)
    print(type(resultados))
    return resultados, nombre_filtro


    