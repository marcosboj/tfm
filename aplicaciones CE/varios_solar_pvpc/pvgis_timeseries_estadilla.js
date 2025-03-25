/*
Descripción:
  Desarollado para APPSCRIPT y Spreadsheet de google. 
  API PVGIS con consulta por hora

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: js pvgis_timeseries_estadilla.js
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
*/




function getPVGISDataPLusEstadilla() {
  // Reemplaza con el ID de tu hoja de cálculo
  var spreadsheetId = 'XXXXXXXXXXXXXXXXXXXXXXXXXx';
  
  // Obtener la hoja de cálculo por su ID
  var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');

  // Leer los valores de las celdas donde están latitud, longitud, mountingplace, angle y aspect
  var lat = sheet.getRange('A2').getValue();  // Cambia 'A2' por la celda correcta
  var lon = sheet.getRange('B2').getValue();  // Cambia 'B2' por la celda correcta
  var peakpower = sheet.getRange('C2').getValue();  // 
  var angle = sheet.getRange('D2').getValue();  // Ángulo de inclinación
  var aspect = sheet.getRange('E2').getValue();  // Orientación (sur, este, oeste, etc.)


  // Generar la URL con los valores obtenidos
  var url = `https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=${lat}&lon=${lon}&angle=${angle}&aspect=${aspect}&peakpower=${peakpower}&startyear=2023&endyear=2023&loss=14&pvcalculation=1&outputformat=json`;
  
  // Realizar la solicitud a la API
  var response = UrlFetchApp.fetch(url);
  
  if (response.getResponseCode() === 200) {
    var data = JSON.parse(response.getContentText());
    // Limpiar el rango donde se insertarán los datos
  sheet.getRange('G1:H13').clearContent();


    
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
    //var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
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
      
    // Insertar datos crudos en la hoja de cálculo
  insertRawDataInSheet(extractedData);

  function insertRawDataInSheet(data) {
  var startRow = 2; // Empezamos desde la fila 2 para evitar la cabecera
  var startCol = 30; // Columna G para 'date', H para 'time', I para 'Producción (W)'

  // Encabezados
  sheet.getRange(1, startCol).setValue('date');
  sheet.getRange(1, startCol + 1).setValue('time');
  sheet.getRange(1, startCol + 2).setValue('Producción (W)');

  // Insertar los datos en la hoja
  data.forEach(function (row, index) {
    var dateString = `2023/${row.month.toString().padStart(2, '0')}/${row.day.toString().padStart(2, '0')}`;
    var timeString = `${row.hour}:00`;

    sheet.getRange(startRow + index, startCol).setValue(dateString);
    sheet.getRange(startRow + index, startCol + 1).setValue(timeString);
    sheet.getRange(startRow + index, startCol + 2).setValue(row.power);
  });
  }
  } else {
    Logger.log('Error en la solicitud: ' + response.getResponseCode());
    }
}

    