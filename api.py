"""
api.py — DZ Bus Tracker · Serveur de prédiction ETA
====================================================
Lance avec : uvicorn api:app --host 0.0.0.0 --port 8000

Endpoint principal : POST /predict
Reçoit la position GPS du bus en temps réel (depuis Firebase via l'app)
Retourne l'ETA en minutes vers chaque arrêt restant sur la ligne.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle, math, numpy as np, os
from datetime import datetime

# ── Charger le modèle au démarrage ────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "eta_model_v4.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        PKG = pickle.load(f)
    print(f"✅ Modèle chargé : {PKG.get('version','?')} | "
          f"MAE Aller={PKG['mae_aller_s']/60:.1f}min | "
          f"MAE Retour={PKG['mae_retour_s']/60:.1f}min | "
          f"Date={PKG['date']}")
except FileNotFoundError:
    raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}. "
                       "Place eta_model_v4.pkl dans le même dossier que api.py")

MODEL_ALLER  = PKG["model_aller"]
MODEL_RETOUR = PKG["model_retour"]
FEATURES     = PKG["features"]
ARRETS_ALLER  = PKG["arrets_aller"]
ARRETS_RETOUR = PKG["arrets_retour"]
DIST_TOT_ALLER  = PKG["dist_tot_aller"]
DIST_TOT_RETOUR = PKG["dist_tot_retour"]


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="DZ Bus Tracker — API ETA",
    description="Prédit le temps d'arrivée d'un bus vers chaque arrêt de la Ligne 24",
    version="4.0",
)

# Autoriser les appels depuis l'app React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas de données ────────────────────────────────────────────────────────

class BusPosition(BaseModel):
    """
    Données GPS du bus envoyées par l'app en temps réel.
    Toutes ces valeurs sont disponibles depuis Firebase bus_actifs.
    """
    # Position GPS
    latitude:  float
    longitude: float
    altitude_m: float = 80.0

    # Mouvement
    vitesse_kmh:      float        # calculée depuis les coordonnées GPS
    vitesse_moy_60s:  float = 0.0  # calculée côté app (moyenne 60 dernières secondes)
    acceleration_kmh_s: float = 0.0
    pente_pct:        float = 0.0
    cap_deg:          float = 0.0

    # Progression dans le trajet
    distance_parcourue_m: float    # depuis le terminus de départ

    # Contexte temporel
    heure_decimal: Optional[float] = None   # si None → calculé automatiquement
    jour_semaine:  Optional[int]   = None   # si None → calculé automatiquement

    # Contexte trajet
    direction:            str    # "Aller" ou "Retour"
    temps_arret_cumule_s: float = 0.0

    # Options
    n_arrets_max: Optional[int] = None   # limiter le nombre d'arrêts retournés


class ArretETA(BaseModel):
    nom:     str
    lat:     float
    lon:     float
    eta_min: float   # minutes
    eta_s:   int     # secondes (plus précis)
    dist_km: float   # distance en km depuis le bus


class PredictionResponse(BaseModel):
    direction:   str
    nb_arrets:   int
    arrets:      List[ArretETA]
    bus_position: dict
    modele_date: str
    timestamp:   str


# ── Logique de prédiction ─────────────────────────────────────────────────────

def calculer_eta_arrets(pos: BusPosition) -> List[ArretETA]:
    """
    Pour chaque arrêt devant le bus, construit le vecteur de features
    et appelle le modèle pour obtenir l'ETA direct.
    """
    direction = pos.direction
    if direction not in ("Aller", "Retour"):
        raise ValueError(f"direction doit être 'Aller' ou 'Retour', reçu : '{direction}'")

    arrets   = ARRETS_ALLER   if direction == "Aller" else ARRETS_RETOUR
    dist_tot = DIST_TOT_ALLER if direction == "Aller" else DIST_TOT_RETOUR
    model    = MODEL_ALLER    if direction == "Aller" else MODEL_RETOUR
    dir_enc  = 0              if direction == "Aller" else 1

    # Calculer heure et jour si non fournis
    now = datetime.now()
    h_dec = pos.heure_decimal if pos.heure_decimal is not None \
            else now.hour + now.minute / 60 + now.second / 3600
    jour  = pos.jour_semaine  if pos.jour_semaine  is not None \
            else now.weekday()

    h_sin = math.sin(2 * math.pi * h_dec / 24)
    h_cos = math.cos(2 * math.pi * h_dec / 24)
    pct_bus = pos.distance_parcourue_m / dist_tot * 100

    resultats = []
    for arret in arrets:
        if arret["dist"] <= pos.distance_parcourue_m:
            continue   # arrêt déjà dépassé

        dv = arret["dist"] - pos.distance_parcourue_m   # distance vers cet arrêt

        # Construire le vecteur dans le même ordre que FEATURES
        X = np.array([[
            h_dec, h_sin, h_cos, jour, int(jour >= 5),
            pos.latitude, pos.longitude, pos.altitude_m, pos.cap_deg,
            pos.vitesse_kmh, pos.vitesse_moy_60s,
            pos.acceleration_kmh_s, pos.pente_pct,
            pct_bus, pos.temps_arret_cumule_s, dir_enc,
            dv,                          # dist_vers_arret_m
            arret["dist"] / dist_tot * 100,  # pct_vers_arret
            math.sqrt(dv),               # dist_vers_arret_sqrt
            dv / dist_tot,               # ratio_dist
        ]])

        eta_s = max(0.0, model.predict(X)[0])

        resultats.append(ArretETA(
            nom     = arret["nom"],
            lat     = arret["lat"],
            lon     = arret["lon"],
            eta_min = round(eta_s / 60, 1),
            eta_s   = int(eta_s),
            dist_km = round(dv / 1000, 2),
        ))

    if pos.n_arrets_max:
        resultats = resultats[:pos.n_arrets_max]

    return resultats


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "DZ Bus Tracker — API ETA",
        "version": "4.0",
        "modele":  PKG.get("version", "?"),
        "date_modele": PKG["date"],
        "mae_aller_min":  round(PKG["mae_aller_s"]  / 60, 2),
        "mae_retour_min": round(PKG["mae_retour_s"] / 60, 2),
        "endpoints": {
            "POST /predict": "Prédire l'ETA vers chaque arrêt",
            "GET  /health":  "Vérifier que le serveur est actif",
            "GET  /arrets/{direction}": "Lister les arrêts Aller ou Retour",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/arrets/{direction}")
def get_arrets(direction: str):
    """Retourne la liste des arrêts d'une direction."""
    if direction == "Aller":
        return {"direction": "Aller", "arrets": ARRETS_ALLER}
    elif direction == "Retour":
        return {"direction": "Retour", "arrets": ARRETS_RETOUR}
    else:
        raise HTTPException(status_code=400, detail="direction doit être 'Aller' ou 'Retour'")


@app.post("/predict", response_model=PredictionResponse)
def predict(pos: BusPosition):
    """
    Endpoint principal — appelé par passager.tsx à chaque mise à jour Firebase.

    Exemple de requête depuis l'app :
    {
        "latitude": 36.755,
        "longitude": 5.060,
        "vitesse_kmh": 28.5,
        "distance_parcourue_m": 2550,
        "direction": "Aller",
        "temps_arret_cumule_s": 45
    }

    Réponse :
    {
        "direction": "Aller",
        "nb_arrets": 9,
        "arrets": [
            {"nom": "Arrêt 8", "eta_min": 1.4, "eta_s": 84, "dist_km": 0.36, ...},
            {"nom": "Arrêt 9", "eta_min": 1.7, "eta_s": 102, ...},
            ...
        ]
    }
    """
    try:
        arrets = calculer_eta_arrets(pos)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        direction    = pos.direction,
        nb_arrets    = len(arrets),
        arrets       = arrets,
        bus_position = {
            "latitude":  pos.latitude,
            "longitude": pos.longitude,
            "vitesse_kmh": pos.vitesse_kmh,
            "distance_parcourue_m": pos.distance_parcourue_m,
        },
        modele_date = PKG["date"],
        timestamp   = datetime.now().isoformat(),
    )
