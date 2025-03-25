/*
Descripción:
  Desarollado para APPSCRIPT y Spreadsheet de google. 
  API PVGIS con consulta por hora. Y consulta API esios para ver precio energia

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: js pvgis_timeseries.js
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
*/



function getPVGISDataPLus() {
  // Reemplaza con el ID de tu hoja de cálculo
  var spreadsheetId = 'XXXXXXXXXXXXXXXXXXXXX';
  
  // Obtener la hoja de cálculo por su ID
  var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');

  // Leer los valores de las celdas donde están latitud, longitud, mountingplace, angle y aspect
  var lat = sheet.getRange('A2').getValue();  // Cambia 'A2' por la celda correcta
  var lon = sheet.getRange('B2').getValue();  // Cambia 'B2' por la celda correcta
  var peakpower = sheet.getRange('C2').getValue();  // 
  var angle = sheet.getRange('D2').getValue();  // Ángulo de inclinación
  var aspect = sheet.getRange('E2').getValue();  // Orientación (sur, este, oeste, etc.)


  // Generar la URL con los valores obtenidos
  var url = `https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=${lat}&lon=${lon}&angle=${angle}&aspect=${aspect}&peakpower=${peakpower}&startyear=2020&endyear=2020&loss=14&pvcalculation=1&outputformat=json`;
  
  // Realizar la solicitud a la API
  var response = UrlFetchApp.fetch(url);
  
  if (response.getResponseCode() === 200) {
    var data = JSON.parse(response.getContentText());
    // Limpiar el rango donde se insertarán los datos
  sheet.getRange('G1:H13').clearContent();

  // Obtener la dirección de una celda específica, por ejemplo, A1
  var address = sheet.getRange('A5').getValue();
  
  // Usar el servicio de geocodificación de Google Maps
  var geocoder = Maps.newGeocoder().geocode(address);
  
  // Verificar si hay resultados válidos
  if (geocoder.results && geocoder.results.length > 0) {
    var location = geocoder.results[0].geometry.location;
    var lat = location.lat.toString().replace(",", ".");
    var lon = location.lng.toString().replace(",", ".");
    
    // Colocar las coordenadas en las celdas B1 y C1
    sheet.getRange('B5').setValue(lat);
    sheet.getRange('C5').setValue(lon);
  } else {
    Logger.log("No se encontraron resultados para la dirección: " + address);
  }
  
  var extractedData = [];

  // Loop through the hourly data to extract time and P (power output)
  data.outputs.hourly.forEach(function (entry) {
    var timeString = entry.time;
    var powerOutput = entry.P;

    // Parse the time string to extract month, day, and hour
    //var year = timeString.substring(0, 4);
    var month = timeString.substring(4, 6);
    month = parseInt(month, 10);
    var day = timeString.substring(6, 8);
    var hour = timeString.substring(9, 11);

    // Save the parsed data
    extractedData.push({
      month: month,
      day: day,
      hour: hour,
      power: powerOutput
    });
  });

  //Logger.log(extractedData);
  // Procesar los datos para calcular el promedio por hora para cada mes
  var hourlyAverages = calculateHourlyAverages(extractedData);

  // Insertar los datos en la hoja de cálculo
  insertDataInSheet(hourlyAverages);

  function calculateHourlyAverages(data) {
    var hourlyData = {};

    // Agrupar los datos por mes y hora
    data.forEach(function (row) {
      var key = row.month + "-" + row.hour;
      if (!hourlyData[key]) {
        hourlyData[key] = { sum: 0, count: 0 };
      }
      hourlyData[key].sum += row.power;
      hourlyData[key].count += 1;
    });

    // Calcular el promedio por hora para cada mes
    var averages = [];
    for (var key in hourlyData) {
      var [month, hour] = key.split('-');
      var averagePower = hourlyData[key].sum / hourlyData[key].count;
      averages.push({ month: parseInt(month, 10), hour: parseInt(hour, 10), averagePower: averagePower });
    }

    return averages;
  }

  function insertDataInSheet(hourlyAverages) {
  
    var startRow = 2;
    var startCol = 15;

    // Insertar encabezados
    sheet.getRange(startRow, startCol).setValue('Mes');
    sheet.getRange(startRow, startCol + 1).setValue('Hora');
    sheet.getRange(startRow, startCol + 2).setValue('Producción Promedio (W)');

    // Insertar los datos en filas
    hourlyAverages.forEach(function (row, index) {
      sheet.getRange(startRow + index + 1, startCol).setValue(row.month);
      sheet.getRange(startRow + index + 1, startCol + 1).setValue(row.hour);
      sheet.getRange(startRow + index + 1, startCol + 2).setValue(row.averagePower);
    });
    
  }
  // Procesar los datos para sumar la producción total por mes
  var monthlyTotals = calculateMonthlyTotals(extractedData);

  // Insertar los totales mensuales en la hoja de cálculo
  insertMonthlyTotalsInSheet(monthlyTotals);


  function calculateMonthlyTotals(data) {
    var monthlyData = {};

    // Agrupar y sumar los datos por mes
    data.forEach(function (row) {
      if (!monthlyData[row.month]) {
        monthlyData[row.month] = 0;
      }
      monthlyData[row.month] += row.power;
    });

    // Convertir a un arreglo con las sumas por cada mes (enero a diciembre)
    var totals = [];
    for (var month = 1; month <= 12; month++) {
      totals.push(monthlyData[month] || 0);
    }

    return totals;
  }

  function insertMonthlyTotalsInSheet(monthlyTotals) {

    var startRow = 2;
    var startCol = 13;

    // Encabezados
    sheet.getRange(startRow, startCol).setValue('Mes');
    sheet.getRange(startRow, startCol + 1).setValue('Producción Total (W)');

    // Nombres de los meses
    var monthNames = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"];

    // Insertar los totales mensuales en las celdas
    for (var i = 0; i < 12; i++) {
      sheet.getRange(startRow + i + 1, startCol).setValue(monthNames[i]);
      sheet.getRange(startRow + i + 1, startCol + 1).setValue(monthlyTotals[i]);
    }
  }
  }     
  
  getEsiosData()
  function getEsiosData() {
    var migeoid = 8741;  // Geolocalización (Peninsula)
    var url2 = 'https://api.esios.ree.es/indicators/1001';
    var headers = {
      'Accept': 'application/json; application/vnd.esios-api-v2+json',
      'Content-Type': 'application/json',
      //'Host': 'api.esios.ree.es',
      'x-api-key': 'febc0ca9a14e6e7ad173e1303c526da331ad7336ca14626177e1e57d2c8e5f8c'
    };
    
    // Realizar la solicitud a la API
    var response2 = UrlFetchApp.fetch(url2, { 'headers': headers });
    
    // Si la solicitud fue exitosa
    if (response2.getResponseCode() === 200) {
      var jsonData = JSON.parse(response2.getContentText());
      
      // Obtener la lista de valores
      var valores = jsonData.indicator.values;
      
      // Filtrar los valores por geo_id
      var valores_geoid = valores.filter(function(val) {
        return val.geo_id === migeoid;
      });
      
      // Extraer las horas y precios
      var horasPrecios = valores_geoid.map(function(val) {
        var date = new Date(val.datetime);
        var hora = Utilities.formatDate(date, Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm');
        var hora0 = Utilities.formatDate(date, Session.getScriptTimeZone(), 'HH');  // Solo la hora
        var precio = val.value;
        return { 'hora': hora, 'precio': precio , 'hora0': hora0};
      });
      
      // Pegarlo en la hoja de cálculo
      
      
      sheet.getRange(2, 19).setValue('Fecha y Hora');
      sheet.getRange(2, 20).setValue('Precio');
      sheet.getRange(2, 21).setValue('Hora');
      
      for (var i = 0; i < horasPrecios.length; i++) {
        sheet.getRange(i + 3, 19).setValue(horasPrecios[i].hora);
        sheet.getRange(i + 3, 20).setValue(horasPrecios[i].precio);
        sheet.getRange(i + 3, 21).setValue(horasPrecios[i].hora0);
      }

      // Obtener solo los precios para estadísticas
      var precios = valores_geoid.map(function(val) {
        return val.value;
      });
      
      // Calcular valores estadísticos
      var valorMin = Math.min.apply(null, precios);
      var valorMax = Math.max.apply(null, precios);
      var valorMed = median(precios);
      
      // Obtener el valor actual basado en la hora
      var currentHour = new Date().getHours();
      var valorActual = null;
      var bajoMedia = false;
      
      valores_geoid.forEach(function(val) {
        var date = new Date(val.datetime);
        if (date.getHours() === currentHour) {
          valorActual = val.value;
          bajoMedia = valorActual < valorMed;
        }
      });
      
      // Imprimir estadísticas en las celdas
      sheet.getRange(horasPrecios.length + 3, 19).setValue('Valor Actual');
      sheet.getRange(horasPrecios.length + 3, 20).setValue(valorActual);
      sheet.getRange(horasPrecios.length + 4, 19).setValue('Valor Max');
      sheet.getRange(horasPrecios.length + 4, 20).setValue(valorMax);
      sheet.getRange(horasPrecios.length + 5, 19).setValue('Valor Min');
      sheet.getRange(horasPrecios.length + 5, 20).setValue(valorMin);
      sheet.getRange(horasPrecios.length + 6, 19).setValue('Mediana');
      sheet.getRange(horasPrecios.length + 6, 20).setValue(valorMed);
      sheet.getRange(horasPrecios.length + 7, 19).setValue('Bajo la Media');
      sheet.getRange(horasPrecios.length + 7, 20).setValue(bajoMedia);
    } else {
      Logger.log('Error en la solicitud: ' + response2.getResponseCode());
    }
      // Función para calcular la mediana
    function median(values) {
      values.sort(function(a, b) { return a - b; });
      var half = Math.floor(values.length / 2);
      
      if (values.length % 2)
        return values[half];
      
      return (values[half - 1] + values[half]) / 2.0;
    }



    
  }
}