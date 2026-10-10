# Documentación (ES): Entrena y prueba un modelo de forma reproducible. Ejecutar: PYTHONHASHSEED=0 python src/train.py --seed 42
"""Train and test a model reproducibly.  Run: PYTHONHASHSEED=0 python src/train.py --seed 42"""
# Módulos: argumentos de línea de comandos, hash, JSON y semillas de Python
import argparse, hashlib, json, random
# NumPy y pandas para datos numéricos y tablas
import numpy as np, pandas as pd
# Modelo de regresión logística
from sklearn.linear_model import LogisticRegression
# Validación cruzada estratificada y su evaluación
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Encadena pasos en un Pipeline
from sklearn.pipeline import make_pipeline
# Escalador que estandariza las variables
from sklearn.preprocessing import StandardScaler

# Fija las semillas de las librerías de azar
def set_seed(seed):
    # Semilla de Python
    random.seed(seed)        # Python
    # Semilla de NumPy
    np.random.seed(seed)     # NumPy  (with PyTorch, also: torch.manual_seed(seed))

# Función principal: entrena y evalúa con una semilla y una ruta de datos
def main(seed=42, data_path="data/breast_cancer.csv"):
    # Fija las semillas antes de cualquier proceso aleatorio
    set_seed(seed)
    # Lee el CSV de datos
    df = pd.read_csv(data_path)
    # Separa variables (X) y etiqueta "target" (y)
    X, y = df.drop(columns="target").values, df["target"].values
    # Pipeline: el escalador aprende solo de cada partición de entrenamiento
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))   # no leakage / sin fuga
    # Validación cruzada de 5 particiones, barajada con la semilla
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Exactitud de cada una de las 5 particiones
    scores = cross_val_score(model, X, y, cv=cv)
    # Devuelve semilla, huella MD5 de los datos y media ± DE de la exactitud
    return {"seed": seed,
            # Clave data_md5: huella MD5 del archivo de datos
            "data_md5": hashlib.md5(open(data_path, "rb").read()).hexdigest(),
            # Clave accuracy_mean: exactitud media redondeada a 5 decimales
            "accuracy_mean": round(float(scores.mean()), 5),
            # Clave accuracy_sd: desviación estándar redondeada a 5 decimales; cierra el diccionario
            "accuracy_sd": round(float(scores.std(ddof=1)), 5)}

# Solo se ejecuta cuando el archivo se lanza como script
if __name__ == "__main__":
    # Define el argumento --seed (entero, por defecto 42) e imprime el resultado como JSON
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    # Lee la semilla indicada en la línea de comandos, ejecuta main e imprime el resultado en JSON
    print(json.dumps(main(ap.parse_args().seed)))
