import requests
import xmltodict
from urllib.parse import urlencode
# Datos de la consulta
provincia = "ZARAGOZA"
municipio = "ZARAGOZA"
sigla = "CL"  # Tipo de vía
nombrevia = "PALOMAR, ALEJANDRO (DR.)"
numero = 25
bloque = ""  # Siempre incluir, aunque esté vacío
escalera = ""  # Siempre incluir, aunque esté vacío
planta = "1"
puerta = "IZ"

# Construcción de la URL completa
base_url = "http://ovc.catastro.meh.es/ovcservweb/OVCSWLocalizacionRC/OVCCallejero.asmx/Consulta_DNPLOC"
params = {
    "Provincia": provincia,
    "Municipio": municipio,
    "Sigla": sigla,
    "Calle": nombrevia,
    "Numero": str(numero),
    "Bloque": bloque,  # Incluye siempre el parámetro
    "Escalera": escalera,  # Incluye siempre el parámetro
    "Planta": planta,
    "Puerta": puerta
}

# Generar URL manualmente
url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
# Codificar los parámetros y generar la URL completa
encoded_params = urlencode(params)
url = f"{base_url}?{encoded_params}"

try:
    # Realizar la solicitud HTTP
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    # Parsear el XML a un diccionario
    data = xmltodict.parse(response.content)

    # Extraer información relevante
    inmueble = data['consulta_dnp']['bico']['bi']
    referencia_catastral = f"{inmueble['idbi']['rc']['pc1']}{inmueble['idbi']['rc']['pc2']}{inmueble['idbi']['rc']['car']}"
    direccion = inmueble['ldt']
    uso = inmueble['debi']['luso']
    superficie = inmueble['debi']['sfc']
    participacion = inmueble['debi']['cpt']
    antiguedad = inmueble['debi']['ant']
    vivienda = next((c for c in data['consulta_dnp']['bico']['lcons']['cons'] if c['lcd'] == "VIVIENDA"), None)
    superficie_vivienda = vivienda['dfcons']['stl'] if vivienda else "No especificado"
    elementos_comunes = next((c for c in data['consulta_dnp']['bico']['lcons']['cons'] if c['lcd'] == "ELEMENTOS COMUNES"), None)
    superficie_comunes = elementos_comunes['dfcons']['stl'] if elementos_comunes else "No especificado"

    # Mostrar resultados
    print(f"URL de la consulta: {url}")
    print("\nResumen visual:")
    print(f"Referencia Catastral: {referencia_catastral}")
    print(f"Dirección: {direccion}")
    print(f"Uso: {uso}")
    print(f"Superficie Total: {superficie} m²")
    print(f"  - Vivienda: {superficie_vivienda} m²")
    print(f"  - Elementos Comunes: {superficie_comunes} m²")
    print(f"Coeficiente de Participación: {participacion}%")
    print(f"Año de Construcción: {antiguedad}")

except Exception as e:
    print(f"Error al realizar la consulta: {e}")
