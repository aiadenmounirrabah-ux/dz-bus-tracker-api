"""
api.py — DZ Bus Tracker · Serveur de prédiction ETA
====================================================
Lance avec : uvicorn api:app --host 0.0.0.0 --port 8000

Endpoint principal : POST /predict
Reçoit la position GPS du bus en temps réel (depuis Firebase via l'app)
Retourne l'ETA en minutes vers chaque arrêt restant sur la ligne,
avec indice de confiance (Random Forest estimators_) et correction trafic
(intelligence collective des bus actifs via Firebase).

Variables d'environnement Render :
  MODEL_PATH           → chemin vers eta_model_v4.pkl (défaut : ./eta_model_v4.pkl)
  FIREBASE_CREDENTIALS → JSON du serviceAccountKey.json (obligatoire pour le trafic)
  FIREBASE_URL         → URL de la Realtime Database (ex: https://xxx-rtdb.firebaseio.com)
"""

import math, os, json, pickle
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Firebase Admin SDK (optionnel — intelligence collective) ──────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

_firebase_ok = False
try:
    _creds_json  = os.getenv("FIREBASE_CREDENTIALS")
    _firebase_url = os.getenv("FIREBASE_URL")
    if _creds_json and _firebase_url:
        try:
            firebase_admin.get_app()          # déjà initialisé (hot-reload)
        except ValueError:
            _cred = credentials.Certificate(json.loads(_creds_json))
            firebase_admin.initialize_app(_cred, {"databaseURL": _firebase_url})
        _firebase_ok = True
        print("✅ Firebase Admin SDK initialisé — intelligence collective activée")
    else:
        print("⚠️  FIREBASE_CREDENTIALS / FIREBASE_URL absentes "
              "— intelligence collective désactivée")
except Exception as _e:
    print(f"⚠️  Firebase non initialisé : {_e}")


# ── Charger le modèle au démarrage ───────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "eta_model_v4.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        PKG = pickle.load(f)
    print(f"✅ Modèle chargé : {PKG.get('version','?')} | "
          f"MAE Aller={PKG['mae_aller_s']/60:.1f}min | "
          f"MAE Retour={PKG['mae_retour_s']/60:.1f}min | "
          f"Date={PKG['date']}")
except FileNotFoundError:
    raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

MODEL_ALLER     = PKG["model_aller"]
MODEL_RETOUR    = PKG["model_retour"]
FEATURES        = PKG["features"]
ARRETS_ALLER    = PKG["arrets_aller"]
ARRETS_RETOUR   = PKG["arrets_retour"]
DIST_TOT_ALLER  = PKG["dist_tot_aller"]
DIST_TOT_RETOUR = PKG["dist_tot_retour"]


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="DZ Bus Tracker — API ETA",
    description="ETA par arrêt avec indice de confiance RF et correction trafic",
    version="5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS (formule exacte de la spec)."""
    R  = 6371000
    p1 = lat1 * math.pi / 180
    p2 = lat2 * math.pi / 180
    dp = (lat2 - lat1) * math.pi / 180
    dl = (lon2 - lon1) * math.pi / 180
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Schémas de données ────────────────────────────────────────────────────────

class BusPosition(BaseModel):
    # Position GPS
    latitude:  float
    longitude: float
    altitude_m: float = 80.0

    # Mouvement
    vitesse_kmh:       float
    vitesse_moy_60s:   float = 0.0
    acceleration_kmh_s: float = 0.0
    pente_pct:         float = 0.0
    cap_deg:           float = 0.0

    # Progression
    distance_parcourue_m: float

    # Contexte temporel
    heure_decimal: Optional[float] = None
    jour_semaine:  Optional[int]   = None

    # Contexte trajet
    direction:            str
    temps_arret_cumule_s: float = 0.0

    # Options
    n_arrets_max: Optional[int] = None

    # Intelligence collective — évite un double appel Firebase depuis l'API
    # L'app envoie déjà les bus actifs qu'elle a récupérés de Firebase.
    # Format attendu : { "bus_id": { "direction": ..., "latitude": ...,
    #                                "longitude": ..., "vitesse": ... }, ... }
    firebase_bus_actifs: Optional[dict] = None


class ArretETA(BaseModel):
    nom:          str
    lat:          float
    lon:          float
    eta_min:      float    # ETA corrigé trafic, en minutes
    eta_s:        int      # ETA corrigé trafic, en secondes
    dist_km:      float
    # ── Amélioration 1 : indice de confiance ──
    eta_min_bas:  float    # borne basse de l'intervalle (en minutes)
    eta_min_haut: float    # borne haute de l'intervalle (en minutes)
    confiance:    str      # "haute" / "moyenne" / "faible"
    ecart_type_s: int      # écart-type brut en secondes


class PredictionResponse(BaseModel):
    direction:   str
    nb_arrets:   int
    arrets:      List[ArretETA]
    bus_position: dict
    modele_date: str
    timestamp:   str
    # ── Amélioration 2 : trafic ──
    coefficient_trafic: float   # 1.0 fluide | 1.15 ralenti | 1.35 embouteillage
    nb_bus_devant:      int
    condition_trafic:   str     # "fluide" / "ralenti" / "embouteillage"


# ── Intelligence collective ───────────────────────────────────────────────────

def get_coefficient_trafic(
    direction: str,
    bus_dist_parcourue: float,
    bus_lat: float,
    bus_lon: float,
    bus_actifs_extern: Optional[dict] = None,
) -> tuple:
    """
    Analyse les bus en avance sur la même direction pour détecter les zones
    de ralentissement et renvoyer un coefficient correcteur.

    Retourne : (coefficient: float, nb_bus_devant: int)
    """
    try:
        # Source des bus actifs : payload de l'app ou Firebase Admin
        if bus_actifs_extern is not None:
            bus_actifs = bus_actifs_extern
            # Accepter aussi une liste [{ id, ... }]
            if isinstance(bus_actifs, list):
                bus_actifs = {str(b.get("id", i)): b for i, b in enumerate(bus_actifs)}
        elif _firebase_ok:
            bus_actifs = firebase_db.reference("bus_actifs").get() or {}
        else:
            return 1.0, 0

        arrets   = ARRETS_ALLER   if direction == "Aller" else ARRETS_RETOUR
        dist_tot = DIST_TOT_ALLER if direction == "Aller" else DIST_TOT_RETOUR

        vitesses_devant = []
        for bus_id, bus in bus_actifs.items():
            if bus.get("direction") != direction:
                continue

            other_lat = bus.get("latitude")
            other_lon = bus.get("longitude")
            if other_lat is None or other_lon is None:
                continue

            # Estimer la distance parcourue de l'autre bus via l'arrêt le plus proche
            min_d = float("inf")
            arret_dist_est = 0.0
            for arret in arrets:
                d = haversine_m(other_lat, other_lon, arret["lat"], arret["lon"])
                if d < min_d:
                    min_d = d
                    arret_dist_est = arret["dist"]

            if arret_dist_est > bus_dist_parcourue:
                vitesses_devant.append(float(bus.get("vitesse", 0) or 0))

        if not vitesses_devant:
            return 1.0, 0

        vit_moy = sum(vitesses_devant) / len(vitesses_devant)
        if vit_moy < 5:
            coeff = 1.35
        elif vit_moy < 15:
            coeff = 1.15
        else:
            coeff = 1.0

        return coeff, len(vitesses_devant)

    except Exception as exc:
        print(f"⚠️  get_coefficient_trafic : {exc}")
        return 1.0, 0


# ── Logique de prédiction ─────────────────────────────────────────────────────

def calculer_eta_arrets(pos: BusPosition):
    """
    Pour chaque arrêt devant le bus :
    1. Construit le vecteur de features
    2. Appelle model.estimators_ pour obtenir l'ETA de chaque arbre RF
    3. Calcule moyenne, écart-type, indice de confiance
    4. Applique le coefficient trafic sur l'ETA final

    Retourne : (List[ArretETA], coefficient_trafic, nb_bus_devant)
    """
    direction = pos.direction
    if direction not in ("Aller", "Retour"):
        raise ValueError(f"direction doit être 'Aller' ou 'Retour', reçu : '{direction}'")

    arrets   = ARRETS_ALLER   if direction == "Aller" else ARRETS_RETOUR
    dist_tot = DIST_TOT_ALLER if direction == "Aller" else DIST_TOT_RETOUR
    model    = MODEL_ALLER    if direction == "Aller" else MODEL_RETOUR
    dir_enc  = 0              if direction == "Aller" else 1

    # Contexte temporel
    now   = datetime.now()
    h_dec = pos.heure_decimal if pos.heure_decimal is not None \
            else now.hour + now.minute / 60 + now.second / 3600
    jour  = pos.jour_semaine  if pos.jour_semaine  is not None \
            else now.weekday()
    h_sin = math.sin(2 * math.pi * h_dec / 24)
    h_cos = math.cos(2 * math.pi * h_dec / 24)
    pct_bus = pos.distance_parcourue_m / dist_tot * 100

    # Coefficient trafic (intelligence collective)
    coeff, nb_bus_devant = get_coefficient_trafic(
        direction,
        pos.distance_parcourue_m,
        pos.latitude,
        pos.longitude,
        pos.firebase_bus_actifs,
    )

    resultats: List[ArretETA] = []
    for arret in arrets:
        if arret["dist"] <= pos.distance_parcourue_m:
            continue   # arrêt déjà dépassé

        dv = arret["dist"] - pos.distance_parcourue_m   # distance vers cet arrêt

        X = np.array([[
            h_dec, h_sin, h_cos, jour, int(jour >= 5),
            pos.latitude, pos.longitude, pos.altitude_m, pos.cap_deg,
            pos.vitesse_kmh, pos.vitesse_moy_60s,
            pos.acceleration_kmh_s, pos.pente_pct,
            pct_bus, pos.temps_arret_cumule_s, dir_enc,
            dv,
            arret["dist"] / dist_tot * 100,
            math.sqrt(dv),
            dv / dist_tot,
        ]])

        # ── Amélioration 1 : prédictions de chaque arbre RF ──────────
        preds_arbres = np.array([tree.predict(X)[0] for tree in model.estimators_])
        eta_s_base   = float(max(0.0, preds_arbres.mean()))
        ecart_type_s = float(preds_arbres.std())

        # Indice de confiance (coefficient de variation)
        cv = ecart_type_s / eta_s_base if eta_s_base > 0 else 0.0
        if cv < 0.15:
            confiance = "haute"
        elif cv < 0.30:
            confiance = "moyenne"
        else:
            confiance = "faible"

        # ── Amélioration 2 : appliquer le coefficient trafic ─────────
        eta_s_corr     = eta_s_base * coeff
        eta_s_bas_corr = max(0.0, (eta_s_base - ecart_type_s)) * coeff
        eta_s_haut_corr = (eta_s_base + ecart_type_s) * coeff

        resultats.append(ArretETA(
            nom          = arret["nom"],
            lat          = arret["lat"],
            lon          = arret["lon"],
            eta_min      = round(eta_s_corr / 60, 1),
            eta_s        = int(eta_s_corr),
            dist_km      = round(dv / 1000, 2),
            eta_min_bas  = round(eta_s_bas_corr / 60, 1),
            eta_min_haut = round(eta_s_haut_corr / 60, 1),
            confiance    = confiance,
            ecart_type_s = int(ecart_type_s),
        ))

    if pos.n_arrets_max:
        resultats = resultats[:pos.n_arrets_max]

    return resultats, coeff, nb_bus_devant


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":       "DZ Bus Tracker — API ETA v5",
        "version":       "5.0",
        "modele":        PKG.get("version", "?"),
        "date_modele":   PKG["date"],
        "mae_aller_min":  round(PKG["mae_aller_s"]  / 60, 2),
        "mae_retour_min": round(PKG["mae_retour_s"] / 60, 2),
        "firebase_trafic": _firebase_ok,
        "endpoints": {
            "POST /predict":             "Prédire l'ETA avec confiance + trafic",
            "GET  /health":              "Vérifier que le serveur est actif",
            "GET  /arrets/{direction}":  "Lister les arrêts Aller ou Retour",
        },
    }


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "firebase":  _firebase_ok,
    }


@app.get("/arrets/{direction}")
def get_arrets(direction: str):
    if direction == "Aller":
        return {"direction": "Aller", "arrets": ARRETS_ALLER}
    elif direction == "Retour":
        return {"direction": "Retour", "arrets": ARRETS_RETOUR}
    else:
        raise HTTPException(status_code=400, detail="direction doit être 'Aller' ou 'Retour'")


@app.post("/predict", response_model=PredictionResponse)
def predict(pos: BusPosition):
    """
    Endpoint principal — appelé par passager.tsx à chaque tap sur un bus.

    Nouveautés v5 :
    - confiance (haute/moyenne/faible) + intervalle [eta_min_bas, eta_min_haut]
      calculés via model.estimators_ (pas de librairie supplémentaire)
    - coefficient_trafic calculé depuis les bus actifs fournis dans le body
      (firebase_bus_actifs) ou lus directement depuis Firebase Admin SDK

    Exemple de requête :
    {
        "latitude": 36.755, "longitude": 5.060,
        "vitesse_kmh": 28.5, "distance_parcourue_m": 2550,
        "direction": "Aller", "temps_arret_cumule_s": 45,
        "firebase_bus_actifs": {
            "Bus_ABC": { "direction": "Aller", "latitude": 36.76,
                         "longitude": 5.07, "vitesse": 12 }
        }
    }
    """
    try:
        arrets, coeff, nb_bus_devant = calculer_eta_arrets(pos)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if coeff >= 1.35:
        condition_trafic = "embouteillage"
    elif coeff > 1.0:
        condition_trafic = "ralenti"
    else:
        condition_trafic = "fluide"

    return PredictionResponse(
        direction    = pos.direction,
        nb_arrets    = len(arrets),
        arrets       = arrets,
        bus_position = {
            "latitude":            pos.latitude,
            "longitude":           pos.longitude,
            "vitesse_kmh":         pos.vitesse_kmh,
            "distance_parcourue_m": pos.distance_parcourue_m,
        },
        modele_date        = PKG["date"],
        timestamp          = datetime.now().isoformat(),
        coefficient_trafic = round(coeff, 2),
        nb_bus_devant      = nb_bus_devant,
        condition_trafic   = condition_trafic,
    )
