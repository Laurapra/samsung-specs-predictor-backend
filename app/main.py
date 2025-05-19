from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import joblib
import pickle
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# Configurar rutas de archivos
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(
  title='Samsung Specs Predictor',
  description='API para predecir precios de dispositivos Samsung basado en sus especificaciones',
  version='1.0.0'
)
# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],  # Permite todos los origenes de desarrollo
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
# Modelos pydantic para la API
class DeviceSpecs(BaseModel):
  internal_storage: int
  storage_ram: int
  battery: int
  primary_camera: str
  display: str
  network: str
  expandible_storage: Optional[float]=None
  
class SimilarDevices(BaseModel):
  name: str
  price: float
  similarity: float

class FeatureContributions(BaseModel):
  base: float
  display: float
  camera: float
  storage: float
  ram: float
  battery: float
  network: float
  expandable: float
  
class PredictionResponse(BaseModel):
  predicted_price: float
  segment: str
  cluster: Optional[int]=None
  feature_contributions: FeatureContributions
  SimilarDevices: List[SimilarDevices]

# Definir valores predeterminados para variables importantes
numeric_cols = ['internal_storage(GB)', 'storage_ram(GB)', 'battery', 'expandable_storage(TB)', 
                'cam1', 'cam2', 'cam3', 'primary_camera_mp', 'network_count']
X_columns = []
price_mean = 0
price_std = 1
model = None
scaler = None
kmeans = None
has_kmeans = False

# Cargar modelos y componentes
try:
    model_path = MODEL_DIR / 'modelo_random_forest.joblib'
    model = joblib.load(model_path)
    
    scaler_path = MODEL_DIR / 'scaler.joblib'
    scaler = joblib.load(scaler_path)

    try:
        kmeans_path = MODEL_DIR / 'kmeans.joblib'
        kmeans = joblib.load(kmeans_path)
        has_kmeans = True
    except Exception as e:
        print(f'Error cargando KMeans: {e}')
        kmeans = None
        has_kmeans = False

    stats_path = MODEL_DIR / 'stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        print('Stats cargadas:', stats)  # Para debug

    price_mean = stats.get('price_mean', 0)
    price_std = stats.get('price_std', 1)
    numeric_cols = stats.get('numeric_cols', numeric_cols)
    X_columns = stats.get('X_columns', [])

    print('MODELOS CARGADOS EXITOSAMENTE')

except Exception as e:
    print('ERROR CARGANDO MODELOS:', e)
  
# Funciones de preprocesamiento
def clasificar_display(display_str):
    display_str = str(display_str).lower()
    if any(x in display_str for x in ['dynamic amoled 2x']):
        return 'premiun_dinamyc_amoled_2x'
    elif any(x in display_str for x in ['dynamic amoled']):
        return 'premiun_dinamyc_amoled'
    elif any(x in display_str for x in ['super amoled']):
        return 'premiun_super_amoled'
    elif any(x in display_str for x in ['qhd', 'quad hd']):
        return 'res_qhd'
    elif any(x in display_str for x in ['full hd']):
        return 'res_full_hd'
    elif any(x in display_str for x in ['hd']):
        return 'res_hd'
    elif any(x in display_str for x in ['pls']):
        return 'panel_pls'
    elif any(x in display_str for x in ['ips']):
        return 'panel_ips'
    elif any(x in display_str for x in ['tft']):
        return 'panel_tft'
    else:
        return 'other'

def preprocess_input(data: dict) -> pd.DataFrame:
    # Convertir a DataFrame
    df = pd.DataFrame([data])
    
    # Renombrar columnas para coincidir con el formato esperado
    column_mapping = {
        'internal_storage': 'internal_storage(GB)',
        'storage_ram': 'storage_ram(GB)',
        'expandible_storage': 'expandable_storage(TB)'  # Corrección del nombre de variable
    }
    
    df = df.rename(columns=column_mapping)
    
    # Extraer información de cámaras
    if 'primary_camera' in df.columns:
        df[['cam1', 'cam2', 'cam3']] = df['primary_camera'].astype(str).str.extract(r'(\d+)[^\d]*(\d+)?[^\d]*(\d+)?')
        for cam in ['cam1', 'cam2', 'cam3']:
            df[cam] = pd.to_numeric(df[cam], errors='coerce').fillna(0)
        
        # Extraer MP principal
        df['primary_camera_mp'] = df['primary_camera'].str.extract(r'(\d+)MP').astype(float)
    
    # Clasificar display
    if 'display' in df.columns:
        df['display_category'] = df['display'].astype(str).apply(clasificar_display)
        # One-hot encoding
        display_dummies = pd.get_dummies(df[['display_category']], drop_first=False)
        df = pd.concat([df, display_dummies], axis=1)
    
    # Procesar network
    if 'network' in df.columns:
        network = df['network'].iloc[0].lower() if not pd.isna(df['network'].iloc[0]) else ""
        df['network_count'] = len(network.split(',')) if network else 0
        df['net_2g'] = 1 if '2g' in network else 0
        df['net_3g'] = 1 if '3g' in network else 0
        df['net_4g'] = 1 if '4g' in network else 0
        df['net_5g'] = 1 if '5g' in network else 0
    
    # Eliminar columnas de texto procesadas
    cols_to_drop = [col for col in ['primary_camera', 'display', 'network'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Añadir columnas price y ratings con 0 si no existen
    if 'price' not in df.columns:
        df['price'] = 0
    if 'ratings' not in df.columns:
        df['ratings'] = 0
    
    # Estandarizar variables numéricas
    numeric_present = [col for col in numeric_cols if col in df.columns]
    df_to_scale = pd.DataFrame()
    
    # Asegurar que tenemos todas las columnas numéricas
    for col in numeric_cols:
        if col in df.columns:
            df_to_scale[col] = df[col]
        else:
            df_to_scale[col] = 0
    
    # Verificar si hay columnas para escalar
    if not df_to_scale.empty and scaler is not None:
        # Aplicar escalado
        scaled_values = scaler.transform(df_to_scale)
        scaled_df = pd.DataFrame(scaled_values, columns=df_to_scale.columns)
        
        # Actualizar columnas escaladas en df
        for col in numeric_cols:
            if col in df.columns:
                df[col] = scaled_df[col]
    
    # Si X_columns está vacío, usar todas las columnas disponibles
    if not X_columns:
        return df
        
    # Asegurar que todas las columnas necesarias estén presentes
    for col in X_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Garantizar mismo orden de columnas que en entrenamiento
    return df[X_columns]

def get_feature_contributions(data_processed: pd.DataFrame) -> Dict[str, float]:
    # Si no hay modelo cargado o X_columns está vacío, devolver valores predeterminados
    if model is None or not X_columns:
        return {
            'base': 0.0,
            'display': 0.0,
            'camera': 0.0,
            'storage': 0.0,
            'ram': 0.0,
            'battery': 0.0,
            'network': 0.0,
            'expandable': 0.0
        }
        
    # Importancia de características del modelo
    feature_importance = pd.DataFrame({
        'feature': X_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calcular contribución aproximada al precio
    base_price = price_mean * 0.15  # Línea base (15% del precio medio)
    
    # Agrupar características relacionadas
    feature_groups = {
        'display': [col for col in X_columns if 'display_category' in col],
        'camera': ['cam1', 'cam2', 'cam3', 'primary_camera_mp'] if 'primary_camera_mp' in X_columns else ['cam1', 'cam2', 'cam3'],
        'storage': ['internal_storage(GB)'],
        'ram': ['storage_ram(GB)'],
        'battery': ['battery'],
        'network': ['network_count', 'net_2g', 'net_3g', 'net_4g', 'net_5g'],
        'expandable': ['expandable_storage(TB)']
    }
    
    # Calcular contribución por grupo
    contributions = {'base': float(base_price)}
    
    # Verificar si 'price' está en data_processed
    if 'price' in data_processed.columns:
        remaining_contribution = price_std * data_processed['price'].iloc[0] + price_mean - base_price
    else:
        remaining_contribution = price_mean - base_price
    
    for group_name, features in feature_groups.items():
        # Solo incluir características presentes
        group_features = [f for f in features if f in feature_importance['feature'].values]
        if not group_features:
            contributions[group_name] = 0.0
            continue
            
        # Calcular importancia total del grupo
        group_importance = feature_importance[feature_importance['feature'].isin(group_features)]['importance'].sum()
        
        # Evitar división por cero
        total_importance = feature_importance['importance'].sum()
        if total_importance > 0:
            # Asignar contribución proporcional
            contributions[group_name] = float(remaining_contribution * (group_importance / total_importance))
        else:
            contributions[group_name] = 0.0
    
    return contributions

def get_similar_devices(specs: dict, predicted_price: float) -> List[Dict[str, Union[str, float]]]:
    # En una implementación real, buscarías en una base de datos
    # Para este ejemplo, usaremos una lógica simplificada basada en el precio
    
    # Catálogo simplificado de dispositivos
    catalog = [
        {"name": "Galaxy S21 Ultra", "price": 1199.99, "specs": {"ram": 12, "display": "Dynamic AMOLED 2X"}},
        {"name": "Galaxy S21", "price": 799.99, "specs": {"ram": 8, "display": "Dynamic AMOLED 2X"}},
        {"name": "Galaxy Note 20", "price": 999.99, "specs": {"ram": 8, "display": "Super AMOLED Plus"}},
        {"name": "Galaxy A53", "price": 449.99, "specs": {"ram": 6, "display": "Super AMOLED"}},
        {"name": "Galaxy A32", "price": 299.99, "specs": {"ram": 4, "display": "TFT LCD"}},
        {"name": "Galaxy M52", "price": 399.99, "specs": {"ram": 6, "display": "AMOLED"}}
    ]
    
    # Calcular similitud basada en precio y algunas características
    similar_devices = []
    
    for device in catalog:
        # Calcular similitud de precio (peso: 70%)
        price_diff = abs(device["price"] - predicted_price)
        # Evitar división por cero
        if predicted_price > 0:
            price_similarity = max(0, 100 - (price_diff / predicted_price * 100)) * 0.7
        else:
            price_similarity = 0
        
        # Similitud de características (peso: 30%)
        feature_similarity = 0
        
        # RAM
        if device["specs"]["ram"] == specs["storage_ram"]:
            feature_similarity += 15
        elif abs(device["specs"]["ram"] - specs["storage_ram"]) <= 2:
            feature_similarity += 10
        
        # Display
        if specs["display"].lower() in device["specs"]["display"].lower():
            feature_similarity += 15
        elif ("amoled" in specs["display"].lower() and "amoled" in device["specs"]["display"].lower()):
            feature_similarity += 10
        
        # Similitud total
        total_similarity = price_similarity + feature_similarity
        
        similar_devices.append({
            "name": device["name"],
            "price": device["price"],
            "similarity": round(total_similarity, 1)
        })
    
    # Ordenar por similitud y tomar los 3 más similares
    similar_devices.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_devices[:3]

# Endpoint de bienvenida
@app.get("/")
def read_root():
  return {'message': 'Bienvenido a la API de Samsung Specs Predictor by DevSharksBQ'}

# Endpoint de predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(specs: DeviceSpecs):
  try:
    # Convertir los datos a diccionario
    input_data = specs.dict()
    
    # Corregir el nombre de la variable si es necesario
    if 'expandible_storage' in input_data and 'expandable_storage' not in input_data:
        input_data['expandable_storage'] = input_data.pop('expandible_storage')
        
    # Preprocesar los datos
    processed_data = preprocess_input(input_data)
    
    # Verificar si el modelo está cargado
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
        
    # Realizar la predicción
    prediction = model.predict(processed_data)[0]
    # Desescalar la predicción
    predicted_price = prediction * price_std + price_mean
    
    # Determinar el segmento
    if predicted_price >= price_mean + 0.5 * price_std:
      segment = 'Premium'
    elif predicted_price <= price_mean - 0.5 * price_std:
      segment = 'Económico'
    else:
      segment = 'Medio'
    
    # Obtener el cluster
    cluster = None
    if has_kmeans and kmeans is not None:
      try:
        # Me aseguro que no tenga NaN
        data_for_cluster = processed_data.fillna(0)
        cluster = int(kmeans.predict(data_for_cluster)[0])
      except Exception as e:
          print(f"Error al predecir cluster: {e}")
          
    # Calcular contribuciones
    contributions_dict = get_feature_contributions(processed_data)
    contributions = FeatureContributions(**contributions_dict)
    
    # Obtengo dispositivos similares
    similar_devices = [
      SimilarDevices(**device)
      for device in get_similar_devices(input_data, predicted_price)
    ]
    # Preparar la respuesta
    response = PredictionResponse(
      predicted_price = round(float(predicted_price), 2),
      segment = segment,
      cluster = cluster,
      feature_contributions = contributions,
      SimilarDevices = similar_devices  # Corregido: usar SimilarDevices en lugar de similar_devicess
    )
    return response
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
@app.get("/predict-get")
def predict_get(
    internal_storage: int, 
    storage_ram: int, 
    battery: int, 
    primary_camera: str, 
    display: str, 
    network: str, 
    expandible_storage: Optional[float] = None
):
    """Endpoint para predecir usando el método GET con query params"""
    specs = DeviceSpecs(
        internal_storage=internal_storage,
        storage_ram=storage_ram,
        battery=battery,
        primary_camera=primary_camera,
        display=display,
        network=network,
        expandible_storage=expandible_storage
    )
    return predict(specs)

@app.get("/status")
def check_status():
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "kmeans_loaded": has_kmeans,
        "model_dir": str(MODEL_DIR),
        "files_found": [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib') or f.endswith('.pkl')]
    }

#ruta de pruebas
@app.get("/predict-demo")
def predict_demo():
    """Endpoint de ejemplo para probar con GET"""
    sample_specs = DeviceSpecs(
        internal_storage=128,
        storage_ram=8,
        battery=5000,
        primary_camera="108MP + 12MP + 5MP",
        display="Dynamic AMOLED 2X",
        network="5G, 4G, 3G, 2G",
        expandible_storage=1.0
    )
    return predict(sample_specs)

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)