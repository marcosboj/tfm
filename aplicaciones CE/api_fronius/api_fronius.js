/*
Descripción:
  Desarollado para APPSCRIPT y Spreadsheet de google. 
  API de fronius para consultar producción.

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: js api_fronius.js
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
*/


function obtenerInstantaeneoPVSystem() {
  // URL base de la API
  var BASE_URL = "https://api.solarweb.com/swqapi";

  // Reemplaza con tu pvSystemId
  var pvSystemId = "XXXXXXXXXXXXXxx";
  // Claves de autenticación (sustituye con tus valores reales)
  var ACCESS_KEY_ID = "XXXXXXXXXXXXXXXXXX";
  var ACCESS_KEY_VALUE = "XXXXXXXXXXXXXXXXXX"; 

  // Rango de fechas (formato: yyyy-MM-ddTHH:mm:ssZ)
  var fromDate = "2025-03-13T00:00:00Z";  // Ajusta la fecha inicial
  var toDate = "2025-03-13T23:59:59Z";    // Ajusta la fecha final

  // Endpoint con los parámetros en la URL
  var ENDPOINT = "/pvsystems/" + pvSystemId + "/flowdata";
  //var queryParams = "?from=" + encodeURIComponent(fromDate) + "&to=" + encodeURIComponent(toDate);
  var queryParams = "?timezone=local"
  var fullUrl = BASE_URL + ENDPOINT + queryParams;



  // Configurar los encabezados
  var headers = {
    "AccessKeyId": ACCESS_KEY_ID,
    "AccessKeyValue": ACCESS_KEY_VALUE,
    "Content-Type": "application/json"
  };

  // Configurar la solicitud
  var options = {
    "method": "get",
    "headers": headers,
    "muteHttpExceptions": true  // Evita que se detenga por errores HTTP
  };

  // Realizar la solicitud GET a la API
  var response = UrlFetchApp.fetch(fullUrl, options);

  // Obtener el código de respuesta
  var statusCode = response.getResponseCode();

  // Manejar la respuesta
  if (statusCode == 200) {
    var data = response.getContentText();
    Logger.log("Respuesta de la API: " + data);
    
    // Si quieres guardar los datos en una hoja de cálculo, descomenta esta línea
    // guardarEnGoogleSheets(data);
  } else {
    Logger.log("Error " + statusCode + ": " + response.getContentText());
  }
}

function obtenerHistorialCompletoPVSystem() {
  var BASE_URL = "https://api.solarweb.com/swqapi";
  // Reemplaza con tu pvSystemId
  var pvSystemId = "740f92fa-7fc6-4596-aa64-2554a698cd20";
  // Claves de autenticación (sustituye con tus valores reales)
  var ACCESS_KEY_ID = "FKIA721BECD157854FFFAF13FDA7EF825040";
  var ACCESS_KEY_VALUE = "1f10ae35-f420-4cef-8cbd-56b54b5a61fa"; 

  // Definir el rango de fechas
  var fromDate = "2025-03-13T00:00:00Z";
  var toDate = "2025-03-13T23:59:59Z";

  // Parámetros de paginación
  var limit = 50;  // Número de registros por solicitud
  var offset = 0;
  var allData = [];


  var headers = {
    "AccessKeyId": ACCESS_KEY_ID,
    "AccessKeyValue": ACCESS_KEY_VALUE,
    "Content-Type": "application/json"
  };

  while (true) {
    var queryParams = "?from=" + encodeURIComponent(fromDate) + 
                      "&to=" + encodeURIComponent(toDate) +
                      "&offset=" + offset +
                      "&limit=" + limit;

    var fullUrl = BASE_URL + "/pvsystems/" + pvSystemId + "/histdata" + queryParams;

    var options = {
      "method": "get",
      "headers": headers,
      "muteHttpExceptions": true
    };

    Logger.log("Consultando API: " + fullUrl);

    var response = UrlFetchApp.fetch(fullUrl, options);
    var statusCode = response.getResponseCode();
    var data = JSON.parse(response.getContentText());

    Logger.log("Respuesta API (offset " + offset + "): " + JSON.stringify(data, null, 2));

    if (statusCode == 200) {
      if (data.data && data.data.length > 0) {
        allData = allData.concat(data.data);  // Agregar datos a la lista
      } else {
        Logger.log("No se encontraron más datos.");
        break;
      }

      // Manejo seguro de paginación
      if (data.links && data.links.next) {
        offset += limit;
      } else {
        break;
      }
    } else {
      Logger.log("Error " + statusCode + ": " + response.getContentText());
      break;
    }
  }

  Logger.log("Total de datos obtenidos: " + allData.length);

  if (allData.length > 0) {
    guardarEnGoogleSheets(allData);
  } else {
    Logger.log("No se encontraron datos para guardar.");
  }
}

function guardarEnGoogleSheets(data) {
  var spreadsheetId = '1yynYGPxGtd_XW7e-lfq-k0KrWdlv30I0_VfqHWukuxk';
  var spreadsheet = SpreadsheetApp.openById(spreadsheetId);
  var sheet = spreadsheet.getSheetByName('Datos');

  if (!sheet) {
    Logger.log("ERROR: No se encontró la hoja 'Datos'.");
    return;
  }

  sheet.clear();

  if (data.length == 0) {
    Logger.log("No hay datos para guardar.");
    return;
  }

  // Crear lista de encabezados
  var headers = ["logDateTime", "logDuration"];
  var channelNames = [];
  
  // Obtener nombres de los canales (columna dinámica)
  if (data[0] && data[0].channels) {
    data[0].channels.forEach(channel => {
      channelNames.push(channel.channelName);
    });
  }

  headers = headers.concat(channelNames);
  sheet.appendRow(headers);

  // Escribir los datos en filas
  data.forEach(function(row) {
    var values = [row.logDateTime, row.logDuration];

    row.channels.forEach(channel => {
      values.push(channel.value);
    });

    sheet.appendRow(values);
  });

  Logger.log("Datos guardados en Google Sheets.");
}
