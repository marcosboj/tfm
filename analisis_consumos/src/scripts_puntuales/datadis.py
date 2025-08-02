def datadis(token,NIF, CUPS, n_supplie, nombre_apellidos):
    
    import requests
    import json
    import pandas as pd
    from datetime import datetime
    from dateutil.relativedelta import relativedelta


    #### Suplies
    url_get_suplies = 'https://datadis.es/api-private/api/get-supplies'
    headers_all = {
        'Authorization': f'Bearer {token}'}

    # Configurar las cabeceras con el token de autenticación
    params_suplies = {
        'authorizedNif': NIF
    }

    response_suplies = requests.get(url_get_suplies, headers=headers_all, params=params_suplies )
    suplies_str = response_suplies.text
    print(suplies_str)
    suplies= json.loads(suplies_str)

    #### Contract
    url_get_contract = 'https://datadis.es/api-private/api/get-contract-detail'

    # Configurar las cabeceras con el token de autenticación
    params_contract = {
        'authorizedNif': NIF,
        'cups': CUPS,
        'distributorCode': int(suplies[n_supplie]['distributorCode'])
    }

    response_contract = requests.get(url_get_contract, headers= headers_all ,params=params_contract)
    contract_str = response_contract.text
    print(contract_str)
    contract = json.loads(contract_str)

    end_date = datetime.today().strftime('%Y/%m')
    #end_date = '2024/06'
    start_date = datetime.strptime(suplies[n_supplie]['validDateFrom'], '%Y/%m/%d').strftime('%Y/%m')
    #start_date = datetime.strptime(contract[0]['startDate'], '%Y/%m/%d').strftime('%Y/%m')
    #start_date = datetime.strptime(contract[0]['dateOwner'][0]['startDate'], '%Y/%m/%d').strftime('%Y/%m')
    
    #start_date = datetime.strptime(contract[0]['dateOwner'][0]['startDate'], '%Y-%m-%d').strftime('%Y/%m')
    two_years_ago = (datetime.now() - relativedelta(years=1, months=11)).strftime('%Y/%m')
    if start_date < two_years_ago:
    # Si start_date es anterior, ajusta a dos años atrás
        start_date = two_years_ago
    else:
    # Si no, conserva start_date
        start_date = start_date
    end_date="2025/06"   
    print(end_date)
    start_date ="2024/12"
    print(start_date)

    #### Consumption
    url_get_consumption = 'https://datadis.es/api-private/api/get-consumption-data'
    # Configurar las cabeceras con el token de autenticación
    params_consumption = {
        'authorizedNif': NIF,
        'cups':CUPS,
        'distributorCode': int(suplies[n_supplie]['distributorCode']),
        'startDate':start_date,
        'endDate':end_date,
        'measurementType' : '0',
        'pointType':suplies[n_supplie]['pointType']
    }

    response_consumption = requests.get(url_get_consumption, headers=headers_all, params=params_consumption )
    consumption = response_consumption.text
    #print(consumption) 

    url_get_power = 'https://datadis.es/api-private/api/get-max-power'
    params_power = {
        'cups':CUPS,
        'distributorCode': int(suplies[n_supplie]['distributorCode']),
        'startDate':start_date,
        'endDate':end_date,
        'authorizedNif': NIF
    }
    
    # Modificar los encabezados para no aceptar 'gzip' como codificación
    headers_all['Accept-Encoding'] = 'identity'

    response_power = requests.get(url_get_power, headers=headers_all, params=params_power)
    power = response_power.text
    
    #print(suplies)
    #print(contract)
     
    if consumption.startswith('Consulta ya realizada en las últimas 24 horas.'): 
        print(consumption)
        exit() 
    #print(power)
    consumption_py = json.loads(consumption)
    power_py = json.loads(power)

    # Paso 2: Crear un DataFrame de pandas a partir de la lista de diccionarios
    df = pd.DataFrame(consumption_py)
    df_p = pd.DataFrame(power_py)

    # Guardar el DataFrame en un archivo CSV
    file_name = f"fichreos_consumo_y_potencias/{nombre_apellidos}_Consumos_{start_date.replace('/', '-')}_{end_date.replace('/', '-')}_{CUPS}.csv"
    df.to_csv(file_name, index=False,)

    #file_name_p = f"fichreos_consumo_y_potencias/{nombre_apellidos}_Potencias_{start_date.replace('/', '-')}_{end_date.replace('/', '-')}_{CUPS}.csv"
    #df_p.to_csv(file_name_p, index=False)
