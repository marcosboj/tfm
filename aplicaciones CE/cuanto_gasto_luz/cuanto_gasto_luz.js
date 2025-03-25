/*
Descripción:
  Desarollado para APPSCRIPT y Spreadsheet de google. 
  Saber cuanto he gastado en la factura de la luz desde tu ultima factura
Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: js cuanto_gasto_luzs.js
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
*/


function obtenerToken() {
    var url = 'https://datadis.es/nikola-auth/tokens/login';
  
    var payload = {
      'username': 'XXXXXXXXXXX',  // Tu NIF
      'password': 'XXXXXXXXXXX'     // Tu contraseña
    };
  
    var options = {
      'method': 'post',
      // Si NO especificas contentType, Google Apps Script lo manda por defecto
      // como x-www-form-urlencoded, que es lo mismo que hace requests.post(..., data=...).
      'payload': payload,
      'muteHttpExceptions': true
    };
  
    // Llamada a la API
    var response = UrlFetchApp.fetch(url, options);
    
    var status = response.getResponseCode();
    var texto = response.getContentText();
    
    Logger.log('Código de respuesta Login: ' + status);
    Logger.log('Respuesta: ' + texto);
    
    if (status === 200) {
      // Aquí "texto" podría ser el token en texto plano
      // o un JSON con el token. Ajusta según necesites.
      return texto;
    } else {
      Logger.log('Error al obtener el token');
      return null;
    }
  }
  
  
  function obtenerFechaFin() {
    var today = new Date();
    var year = today.getFullYear();
    var month = ('0' + (today.getMonth() + 1)).slice(-2); // Asegura dos dígitos
    return year + '/' + month;
  }
  
  function obtenerFechaInicio() {
    var spreadsheetId = '1ESliqs1IjpQDel6g6I0sdUVntANWSQiV9NEBuniptVw';
    var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Datos');
    var fechaCelda = sheet.getRange('D2').getValue();
  
    Logger.log(fechaCelda)
    ///
    if (fechaCelda instanceof Date && !isNaN(fechaCelda)) {  // Verifica si es una fecha válida
      var year = fechaCelda.getFullYear();
      var month = ('0' + (fechaCelda.getMonth() + 1)).slice(-2);
      return year + '/' + month;
    } else {
      Logger.log('La fecha de inicio no es válida o está vacía.');
      return null;
    }
    
  }
  
  
  function obtenerConsumo(token, NIF, CUPS, startDate, endDate, distributorCode, pointType) {
    // URL base
    var url = 'https://datadis.es/api-private/api/get-consumption-data';
    
    // Construir parámetros
    var params = 'authorizedNif=' + encodeURIComponent(NIF) +
                 '&cups=' + encodeURIComponent(CUPS) +
                 '&startDate=' + encodeURIComponent(startDate) +
                 '&endDate=' + encodeURIComponent(endDate) +
                 '&measurementType=0' +
                 '&pointType=' + encodeURIComponent(pointType) +
                 '&distributorCode=' + encodeURIComponent(distributorCode);
    
    // Cabeceras (con X-Channel y Accept si DataDis lo pide)
    var headers = {
      'Authorization': 'Bearer ' + token,
      'Accept': 'application/json',
      'X-Channel': 'B2C'  // Si la documentación lo exige
    };
    
    // Opciones
    var options = {
      'method': 'get',
      'headers': headers,
      'muteHttpExceptions': true
    };
  
    var fullUrl = url + '?' + params;
    
    // Petición
    var response = UrlFetchApp.fetch(fullUrl, options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    //Logger.log('Código de respuesta: ' + responseCode);
    //Logger.log('Respuesta completa: ' + responseText);
    
    if (responseCode === 200) {
        // Parsear el JSON
        var data = JSON.parse(responseText);
  
        // Obtener la hoja "Consumo" (o "Consumos")
        var spreadsheetId = '1ESliqs1IjpQDel6g6I0sdUVntANWSQiV9NEBuniptVw';
        var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Consumo');
        // Limpia lo anterior si quieres sobreescribir (opcional)
        sheet.clear();
  
        // Cabecera
        sheet.appendRow(['Fecha', 'Hora', 'Consumo (kWh)']);
  
        // Si la respuesta es un único objeto
        if (!Array.isArray(data)) {
          sheet.appendRow([data.date, data.time, data.consumptionKWh]);
        } 
        // Si la respuesta es un array de varios objetos
        else {
          data.forEach(function(item) {
            sheet.appendRow([item.date, item.time, item.consumptionKWh]);
            SpreadsheetApp.flush()
          });
        }
  
      } else {
        // Error en la respuesta
        Logger.log('Error al obtener los datos: ' + responseText);
        Logger.log('Detalles de la respuesta: ' + JSON.stringify(response));
      }
     
  }
  
  function getEsiosData() {
    // 1) Leer la fecha inicial de la celda D2 en la hoja 'Datos'
    var spreadsheetId = '1ESliqs1IjpQDel6g6I0sdUVntANWSQiV9NEBuniptVw';
    var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Datos');
    var fechaInicio = sheet.getRange('D2').getValue(); 
    // Asegúrate de que sea un objeto Date
    if (!(fechaInicio instanceof Date)) {
      Logger.log('La celda D2 no contiene una fecha válida.');
      return;
    }
  
    // 2) Definir la fecha final (hoy)
    //var fechaFin = new Date();  // Ahora mismo
    var fechaFin = sheet.getRange('C4').getValue();
  
    // Opcional: si quieres forzar "00:00" y "23:59" usa:
    // fechaInicio.setHours(0, 0, 0);
    // fechaFin.setHours(23, 59, 0);
  
    // 3) Formatear las fechas como dd-MM-yyyy'T'HH:mm
    var formato = "dd-MM-yyyy'T'HH:mm";
    var inicioStr = Utilities.formatDate(fechaInicio, "Europe/Madrid", formato);
    var finStr    = Utilities.formatDate(fechaFin,   "Europe/Madrid", formato);
  
    // 4) Construir la URL con los query params adecuados
    var url = 'https://api.esios.ree.es/indicators/1001'
            + '?start_date=' + encodeURIComponent(inicioStr)
            + '&end_date='   + encodeURIComponent(finStr)
            + '&groupby=hour';  // Suele usarse para obtener datos horarios
  
    // 5) Headers para la API de ESIOS
    var headers = {
      'Accept': 'application/json; application/vnd.esios-api-v2+json',
      'Content-Type': 'application/json',
      'x-api-key': 'febc0ca9a14e6e7ad173e1303c526da331ad7336ca14626177e1e57d2c8e5f8c' // Sustituye con tu clave real
    };
  
    // Llamada a la API
    var response = UrlFetchApp.fetch(url, {
      'headers': headers,
      'muteHttpExceptions': true
    });
  
    // 6) Procesar la respuesta
    var statusCode = response.getResponseCode();
    Logger.log('Código de respuesta ESIOS: ' + statusCode);
  
    if (statusCode === 200) {
      var jsonData = JSON.parse(response.getContentText());
      
      // Filtrar por geo_id si es necesario. 
      // 8741 = Península, 8742 = Baleares, etc.
      var migeoid = 8741; 
      var valores = jsonData.indicator.values.filter(function(val) {
        return val.geo_id === migeoid;
      });
  
      // 7) Guardar la info en la hoja "Precios"
      var hojaPrecios = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Precios');
      hojaPrecios.clear();
      hojaPrecios.appendRow(['Fecha y Hora', 'Fecha', 'Hora', 'Precio']);
  
      valores.forEach(function(val) {
        // La propiedad "datetime" vendrá en formato ISO, p.e. "2025-02-01T00:00:00.000+01:00"
        var dateObj = new Date(val.datetime);
        var fechaHora = Utilities.formatDate(dateObj, Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm');
        var horaSolo  = Utilities.formatDate(dateObj, Session.getScriptTimeZone(), 'HH');
        var fechaSolo = Utilities.formatDate(dateObj, Session.getScriptTimeZone(), 'yyyy/MM/dd');
        var precio    = val.value; // el valor horario
        hojaPrecios.appendRow([fechaHora, fechaSolo,horaSolo, precio]);
      });
  
      Logger.log('Datos de ESIOS cargados correctamente.');
  
    } else {
      // Manejo de errores
      Logger.log('Error al obtener datos de precios: ' + response.getContentText());
    }
  }
  
  
  function ejecutarTodo() {
    var token = obtenerToken();
    if (!token) {
      Logger.log('No se pudo obtener el token, deteniendo ejecución.');
      return;
    }
    
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Datos');
    var CUPS = sheet.getRange('A2').getValue();
    var NIF = sheet.getRange('B2').getValue();
    
    // Llamadas correctas a las funciones que sí existen
    var startDate = obtenerFechaInicio();//obtenerFechaInicio();  
    var endDate = obtenerFechaFin();       
    
    Logger.log(startDate);
    Logger.log(endDate);
    
    if (!startDate || !endDate) {
      Logger.log('Fechas no válidas, deteniendo ejecución.');
      return;
    }
    
    var distributorCode = "2";
    var pointType = 5;
    obtenerConsumo(token, NIF, CUPS, startDate, endDate, distributorCode, pointType);
    getEsiosData();
    getDatos();
  }
  
  
  function doGet() {
    return HtmlService.createHtmlOutputFromFile('index')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
  }
  
  function getDatos() {
    var spreadsheetId = '1ESliqs1IjpQDel6g6I0sdUVntANWSQiV9NEBuniptVw';
    var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('Datos');
  
    var data = {
  
      "fecha_inicio": sheet.getRange("D2").getDisplayValue(),
      "fecha_fin": sheet.getRange("D3").getDisplayValue(),
      "ultimo_dia_datos": sheet.getRange("D4").getDisplayValue(),
      "coste_ultimo_dia": sheet.getRange("K4").getDisplayValue(),
      "coste_acumulado": sheet.getRange("K6").getDisplayValue(),
      "contador_kWh": sheet.getRange("K11").getDisplayValue(),
      "limite_kWh": sheet.getRange("K12").getDisplayValue(),
      "sobrepasado_limite": sheet.getRange("K13").getValue(),
      "kWh_restantes": sheet.getRange("K14").getDisplayValue()
    };
  
    return data;
  }