function getPVGISDataPLusEstadilla() {
  // Reemplaza con el ID de tu hoja de cálculo
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  
  // Obtener la hoja de cálculo por su ID
  var sheet_pv = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');
  // Obtener la dirección de una celda específica, por ejemplo, A1
  var address = sheet_pv.getRange('A5').getValue();
  
  // Usar el servicio de geocodificación de Google Maps
  var geocoder = Maps.newGeocoder().geocode(address);
  
  // Verificar si hay resultados válidos
  if (geocoder.results && geocoder.results.length > 0) {
    var location = geocoder.results[0].geometry.location;
    var lat = location.lat.toString().replace(",", ".");
    var lon = location.lng.toString().replace(",", ".");
    
    // Colocar las coordenadas en las celdas B1 y C1
    sheet_pv.getRange('B5').setValue(lat);
    sheet_pv.getRange('C5').setValue(lon);
  } else {
    Logger.log("No se encontraron resultados para la dirección: " + address);
  }

  // Leer los valores de las celdas donde están latitud, longitud, mountingplace, angle y aspect
  var lat = sheet_pv.getRange('A2').getValue();  // Cambia 'A2' por la celda correcta
  var lon = sheet_pv.getRange('B2').getValue();  // Cambia 'B2' por la celda correcta
  var peakpower = sheet_pv.getRange('C2').getValue();  // 
  var angle = sheet_pv.getRange('D2').getValue();  // Ángulo de inclinación
  var aspect = sheet_pv.getRange('E2').getValue();  // Orientación (sur, este, oeste, etc.)


  // Generar la URL con los valores obtenidos
  var url = `https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=${lat}&lon=${lon}&angle=${angle}&aspect=${aspect}&peakpower=${peakpower}&startyear=2023&endyear=2023&loss=14&pvcalculation=1&outputformat=json`;
  
  // Realizar la solicitud a la API
  var response = UrlFetchApp.fetch(url);
  
  if (response.getResponseCode() === 200) {
    var data = JSON.parse(response.getContentText());
    // Limpiar el rango donde se insertarán los datos
  sheet_pv.getRange('G1:H13').clearContent();


    
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
    var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
    var sheet_pv = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');
    var startRow = 2;
    var startCol = 15;

    // Insertar encabezados
    sheet_pv.getRange(startRow, startCol).setValue('Mes');
    sheet_pv.getRange(startRow, startCol + 1).setValue('Hora');
    sheet_pv.getRange(startRow, startCol + 2).setValue('Producción Promedio (W)');

    // Insertar los datos en filas
    hourlyAverages.forEach(function (row, index) {
      sheet_pv.getRange(startRow + index + 1, startCol).setValue(row.month);
      sheet_pv.getRange(startRow + index + 1, startCol + 1).setValue(row.hour);
      sheet_pv.getRange(startRow + index + 1, startCol + 2).setValue(row.averagePower);
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
    var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
    var sheet_pv = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');    
    var startRow = 2;
    var startCol = 13;

    // Encabezados
    sheet_pv.getRange(startRow, startCol).setValue('Mes');
    sheet_pv.getRange(startRow, startCol + 1).setValue('Producción Total (W)');

    // Nombres de los meses
    var monthNames = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"];

    // Insertar los totales mensuales en las celdas
    for (var i = 0; i < 12; i++) {
      sheet_pv.getRange(startRow + i + 1, startCol).setValue(monthNames[i]);
      sheet_pv.getRange(startRow + i + 1, startCol + 1).setValue(monthlyTotals[i]);
    }
  }
      
    // Insertar datos crudos en la hoja de cálculo
  insertRawDataInSheet(extractedData);

  function insertRawDataInSheet(data) {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_pv = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');
  var startRow = 2; // Empezamos desde la fila 2 para evitar la cabecera
  var startCol = 30; // Columna G para 'date', H para 'time', I para 'Producción (W)'

  // Encabezados
  sheet_pv.getRange(1, startCol).setValue('date');
  sheet_pv.getRange(1, startCol + 1).setValue('time');
  sheet_pv.getRange(1, startCol + 2).setValue('Producción (W)');

  // Insertar los datos en la hoja
  data.forEach(function (row, index) {
    var dateString = `2023/${row.month.toString().padStart(2, '0')}/${row.day.toString().padStart(2, '0')}`;
    var timeString = `${row.hour}:00`;

    sheet_pv.getRange(startRow + index, startCol).setValue(dateString);
    sheet_pv.getRange(startRow + index, startCol + 1).setValue(timeString);
    sheet_pv.getRange(startRow + index, startCol + 2).setValue(row.power);
  });
  }
  } else {
    Logger.log('Error en la solicitud: ' + response.getResponseCode());
    }
}
function getCurrentAddress() {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_pv = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');
  // Leer la dirección almacenada en la celda A5
  var address = sheet_pv.getRange("A5").getValue();
  
  return address ? address : "Dirección no configurada";
}


function getData() {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');
  var data = sheet_vista.getDataRange().getValues(); // Obtiene todos los datos

  return JSON.stringify(data); // Convierte los datos en formato JSON
}

function getChartImage() {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');
  var charts = sheet_vista.getCharts();
  
  if (charts.length > 0) {
    var chartBlob = charts[0].getAs('image/png'); // Obtiene la imagen del primer gráfico
    var url = "data:image/png;base64," + Utilities.base64Encode(chartBlob.getBytes()); // Convierte en base64
    return url;
  } else {
    return "No se encontró ninguna gráfica en la hoja.";
  }
}
function getTableData() {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');
  var range = sheet_vista.getRange("J1:X40"); // Ajusta el rango según la cantidad de filas a mostrar
  var values = range.getValues(); // Obtiene los valores como una matriz

  var result = [];

  for (var i = 0; i < values.length; i++) {
    var row = [];
    for (var j = 0; j < values[i].length; j++) {
      var cell = values[i][j];
      if (typeof cell === "string" && cell.startsWith("http")) {
        // Si es un link (puede ser una imagen), lo convertimos en una etiqueta <img>
        row.push(`<img src="${cell}" style="max-width:100px; max-height:100px;">`);
      } else {
        row.push(cell); // De lo contrario, solo agregamos el texto
      }
    }
    result.push(row);
  }

  return result;
}
function updateMonthAndPower(month, power) {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');
  
  // Escribir los valores seleccionados en las celdas correspondientes
  sheet_vista.getRange("B1").setValue(month); // Celda donde se guarda el mes seleccionado
  sheet_vista.getRange("F1").setValue(power); // Celda donde se guarda la potencia instalada


  return "Datos actualizados correctamente";
}
function updateAddress(street, number, postalCode, city, province, inclinacion, orientacion) {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');

  // Formar la dirección completa
  //var fullAddress = street + " " + number + ", " + postalCode + ", " + city + ", " + province;
  
  // Escribir la dirección en la celda correspondiente
  sheet_vista.getRange("K2").setValue(street);
  sheet_vista.getRange("L2").setValue(number);
  sheet_vista.getRange("M2").setValue(postalCode);
  sheet_vista.getRange("N2").setValue(city);
  sheet_vista.getRange("O2").setValue(province);
  sheet_vista.getRange("Q2").setValue(inclinacion);
  sheet_vista.getRange("R2").setValue(orientacion);

  // Llamar a la función que obtiene las coordenadas y actualiza la hoja
  getPVGISDataPLusEstadilla();

  return "Dirección actualizada correctamente";
}


function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function getAnalysisData() {
  var spreadsheetId = '1Sce1itOwXvvrDU13JLt0ZfFMD61qbU8VzImJInv8EBg';
  var sheet_vista = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Vista');

  var data = {
    resumen: {
      consumoAnual: parseFloat(sheet_vista.getRange("K7").getValue()) || 0,
      produccionAnual: parseFloat(sheet_vista.getRange("K10").getValue()) || 0,
      consumoHorasSolares: parseFloat(sheet_vista.getRange("L20").getValue()) || 0
    },
    resumenMensual: sheet_vista.getRange("M5:P17").getValues().map(row => 
      row.map(num => (typeof num === "number" && !isNaN(num)) ? num.toFixed(1) : "0.0")
    ),
    repartoProduccion: {
      energiaAutoconsumida: parseFloat(sheet_vista.getRange("L22").getValue()) || 0,
      energiaExcedente: parseFloat(sheet_vista.getRange("L23").getValue()) || 0
    },
    recordatorio: {
      texto: sheet_vista.getRange("J28").getValue() || "Sin datos",
      imagenResumen: sheet_vista.getRange(":Q38").getValue() || "Sin imagen"
    },
    precios: {
      precioCompra: parseFloat(sheet_vista.getRange("N41").getValue()) || 0,
      precioVenta: parseFloat(sheet_vista.getRange("N42").getValue()) || 0,
      costoSinSolar: parseFloat(sheet_vista.getRange("N44").getValue()) || 0,
      costoConSolar: parseFloat(sheet_vista.getRange("N45").getValue()) || 0
    },
    amortizacion: {
      opcionesTamano: [
        "< 5 kW", "5 - 10 kW", "10 - 20 kW",
        "20 - 30 kW", "30 - 50 kW", "50 - 75 kW",
        "75 - 100 kW"
      ],
      tamanoSeleccionado: sheet_vista.getRange("L50").getValue() || "< 5 kW",
      precioPorKW: parseFloat(sheet_vista.getRange("N50").getValue()) || 0,
      precioInicial: parseFloat(sheet_vista.getRange("L51").getValue()) || 0,
      ahorroAnual: parseFloat(sheet_vista.getRange("L52").getValue()) || 0,
      anosAmortizacion: parseFloat(sheet_vista.getRange("L53").getValue()) || 0,
      ahorro30anos: parseFloat(sheet_vista.getRange("L54").getValue()) || 0
    }
  };

  Logger.log("📌 Datos obtenidos:");
  Logger.log(JSON.stringify(data, null, 2)); // Formato JSON para debug

  return data;
}
