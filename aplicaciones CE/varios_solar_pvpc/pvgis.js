/*
Descripción:
  Desarollado para APPSCRIPT y Spreadsheet de google. 
  API PVGIS 
Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: js pvgis.js
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
*/

function getPVGISData() {
    // Reemplaza con el ID de tu hoja de cálculo
    var spreadsheetId = 'XXXXXXXXXXXXXXXXXXXXXXXxx';
    
    // Obtener la hoja de cálculo por su ID
    var sheet = SpreadsheetApp.openById(spreadsheetId).getSheetByName('PV');
  
    // Leer los valores de las celdas donde están latitud, longitud, mountingplace, angle y aspect
    var lat = sheet.getRange('A2').getValue();  // Cambia 'A2' por la celda correcta
    var lon = sheet.getRange('B2').getValue();  // Cambia 'B2' por la celda correcta
    var peakpower = sheet.getRange('C2').getValue();  // 
    var angle = sheet.getRange('D2').getValue();  // Ángulo de inclinación
    var aspect = sheet.getRange('E2').getValue();  // Orientación (sur, este, oeste, etc.)
  
    // Generar la URL con los valores obtenidos
    var url = `https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=${lat}&lon=${lon}&angle=${angle}&aspect=${aspect}&peakpower=${peakpower}&loss=14&`;
    
    // Realizar la solicitud a la API
    var response = UrlFetchApp.fetch(url);
    
    if (response.getResponseCode() === 200) {
      var data = JSON.parse(response.getContentText());
      Logger.log(data);  // Imprimir los resultados
    } else {
      Logger.log('Error en la solicitud: ' + response.getResponseCode());
    }
  
  
   
    // Limpiar el rango donde se insertarán los datos
    sheet.getRange('G1:H13').clearContent();
  
    // Insertar encabezados
    sheet.getRange('G1').setValue('Mes');
    sheet.getRange('H1').setValue('E_m (kWh)');
  
    // Recorrer el JSON y pegar los datos en la hoja de cálculo
    var monthlyData = data.outputs.monthly.fixed;
    for (var i = 0; i < monthlyData.length; i++) {
      var month = monthlyData[i].month;
      var E_m = monthlyData[i].E_m;
  
      sheet.getRange(i + 2, 7).setValue(month); // Columna G (Mes)
      sheet.getRange(i + 2, 8).setValue(E_m);   // Columna H (E_m)
    }
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
  
  }