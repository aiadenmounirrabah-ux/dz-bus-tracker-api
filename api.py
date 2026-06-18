"""
api.py — DZ Bus Tracker · Serveur de prédiction ETA v5.0
=========================================================
Lance avec : uvicorn api:app --host 0.0.0.0 --port $PORT

Nouveautés v5.0 :
  - Algorithme : XGBoost (au lieu de Random Forest)
  - 3 modèles : model_eta + model_bas + model_haut (régression quantile)
  - Intervalles de confiance [10% — 90%]
  - Indice de confiance : haute / moyenne / faible
  - Intelligence collective des bus via Firebase
  - Week-end algérien corrigé : vendredi(4) + samedi(5)

Variables d'environnement Render :
  MODEL_PATH           → chemin vers eta_model_v5.pkl (défaut : ./eta_model_v5.pkl)
  FIREBASE_CREDENTIALS → JSON du serviceAccountKey.json
  FIREBASE_URL         → URL de la Realtime Database
  PYTHON_VERSION       → 3.11.9
"""

import math, os, json, pickle
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Firebase Admin SDK (intelligence collective) ──────────────────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

_firebase_ok = False
try:
    _creds_json   = os.getenv("FIREBASE_CREDENTIALS")
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


# ── Charger le modèle XGBoost au démarrage ────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "eta_model_v5.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        PKG = pickle.load(f)
    print(f"✅ Modèle chargé : {PKG.get('version','?')} | "
          f"MAE={PKG['mae_global_s']/60:.2f}min | "
          f"R²={PKG.get('r2_global',0):.4f} | "
          f"Algorithme={PKG.get('algorithme','XGBoost')} | "
          f"Date={PKG['date']}")
except FileNotFoundError:
    raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}. "
                       "Place eta_model_v5.pkl dans le même dossier que api.py")

MODEL_ETA     = PKG["model_eta"]     # prédiction principale
MODEL_BAS     = PKG["model_bas"]     # borne basse (percentile 10%)
MODEL_HAUT    = PKG["model_haut"]    # borne haute (percentile 90%)
FEATURES      = PKG["features"]
ARRETS_ALLER  = PKG["arrets_aller"]
ARRETS_RETOUR = PKG["arrets_retour"]
DIST_TOT_ALLER  = PKG["dist_tot_aller"]
DIST_TOT_RETOUR = PKG["dist_tot_retour"]


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="DZ Bus Tracker — API ETA",
    description=(
        "Prédiction ETA par arrêt avec XGBoost + intervalles de confiance. "
        "Intelligence collective des bus via Firebase."
    ),
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
    """Distance en mètres entre deux points GPS."""
    R  = 6371000
    p1 = lat1 * math.pi / 180
    p2 = lat2 * math.pi / 180
    dp = (lat2 - lat1) * math.pi / 180
    dl = (lon2 - lon1) * math.pi / 180
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_confiance(eta_s: float, bas_s: float, haut_s: float) -> str:
    """
    Calcule l'indice de confiance depuis la largeur de l'intervalle.
    Plus l'intervalle est étroit par rapport à l'ETA, plus on est sûr.
    """
    if eta_s <= 0:
        return "faible"
    largeur = max(0.0, haut_s - bas_s)
    cv = largeur / (2 * eta_s)   # coefficient de variation normalisé
    if cv < 0.20:
        return "haute"
    elif cv < 0.40:
        return "moyenne"
    else:
        return "faible"


# ── Schémas de données ────────────────────────────────────────────────────────

class BusPosition(BaseModel):
    """
    Données GPS du bus envoyées par l'app en temps réel.
    Toutes ces valeurs sont disponibles depuis Firebase bus_actifs.
    """
    # Position GPS
    latitude:   float
    longitude:  float
    altitude_m: float = 80.0

    # Mouvement
    vitesse_kmh:        float
    vitesse_moy_60s:    float = 0.0
    acceleration_kmh_s: float = 0.0
    pente_pct:          float = 0.0
    cap_deg:            float = 0.0

    # Progression dans le trajet
    distance_parcourue_m: float

    # Contexte temporel (calculé automatiquement si non fourni)
    heure_decimal: Optional[float] = None
    jour_semaine:  Optional[int]   = None

    # Contexte trajet
    direction:            str    # "Aller" ou "Retour"
    temps_arret_cumule_s: float = 0.0

    # Options
    n_arrets_max: Optional[int] = None   # limiter le nombre d'arrêts retournés

    # Intelligence collective
    # L'app envoie les bus actifs déjà disponibles pour éviter un double appel Firebase
    # Format : { "bus_id": { "direction": ..., "latitude": ..., "longitude": ..., "vitesse": ... } }
    firebase_bus_actifs: Optional[dict] = None


class ArretETA(BaseModel):
    nom:     str
    lat:     float
    lon:     float
    # ETA principal
    eta_min: float   # minutes (corrigé trafic)
    eta_s:   int     # secondes (corrigé trafic)
    dist_km: float   # distance en km depuis le bus
    # Intervalle de confiance (régression quantile XGBoost)
    eta_min_bas:  float   # borne basse — percentile 10%
    eta_min_haut: float   # borne haute — percentile 90%
    confiance:    str     # "haute" / "moyenne" / "faible"


class PredictionResponse(BaseModel):
    direction:   str
    nb_arrets:   int
    arrets:      List[ArretETA]
    bus_position: dict
    modele_date: str
    timestamp:   str
    # Intelligence collective
    coefficient_trafic: float   # 1.0 fluide | 1.15 ralenti | 1.35 embouteillage
    nb_bus_devant:      int     # nombre de bus analysés
    condition_trafic:   str     # "fluide" / "ralenti" / "embouteillage"


# ── Intelligence collective des bus ──────────────────────────────────────────

def get_coefficient_trafic(
    direction: str,
    bus_dist_parcourue: float,
    bus_lat: float,
    bus_lon: float,
    bus_actifs_extern: Optional[dict] = None,
) -> tuple:
    """
    Analyse les bus en avance sur la même direction pour détecter les zones
    de ralentissement et retourner un coefficient correcteur.

    Principe (innovation propre au projet) :
    Si un bus devant roule à < 5 km/h → embouteillage → ETA × 1.35
    Si un bus devant roule à < 15 km/h → ralenti → ETA × 1.15
    Sinon → trafic fluide → ETA × 1.0

    Retourne : (coefficient: float, nb_bus_devant: int)
    """
    try:
        # Source des bus actifs : payload de l'app ou Firebase Admin
        if bus_actifs_extern is not None:
            bus_actifs = bus_actifs_extern
            if isinstance(bus_actifs, list):
                bus_actifs = {str(b.get("id", i)): b
                              for i, b in enumerate(bus_actifs)}
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

            # Estimer la position de l'autre bus via l'arrêt le plus proche
            min_d = float("inf")
            arret_dist_est = 0.0
            for arret in arrets:
                d = haversine_m(other_lat, other_lon, arret["lat"], arret["lon"])
                if d < min_d:
                    min_d = d
                    arret_dist_est = arret["dist"]

            # Garder seulement les bus qui sont DEVANT notre bus
            if arret_dist_est > bus_dist_parcourue:
                vitesses_devant.append(float(bus.get("vitesse", 0) or 0))

        if not vitesses_devant:
            return 1.0, 0

        vit_moy = sum(vitesses_devant) / len(vitesses_devant)
        if vit_moy < 5:
            coeff = 1.35   # embouteillage
        elif vit_moy < 15:
            coeff = 1.15   # ralenti
        else:
            coeff = 1.0    # fluide

        return coeff, len(vitesses_devant)

    except Exception as exc:
        print(f"⚠️  get_coefficient_trafic : {exc}")
        return 1.0, 0


# ── Logique de prédiction ─────────────────────────────────────────────────────

def calculer_eta_arrets(pos: BusPosition):
    """
    Pour chaque arrêt devant le bus :
    1. Construit le vecteur de features (même ordre que l'entraînement)
    2. Appelle les 3 modèles XGBoost (eta, bas, haut)
    3. Applique le coefficient trafic (intelligence collective)
    4. Calcule l'indice de confiance depuis la largeur de l'intervalle

    Retourne : (List[ArretETA], coefficient_trafic, nb_bus_devant)
    """
    direction = pos.direction
    if direction not in ("Aller", "Retour"):
        raise ValueError(f"direction doit être 'Aller' ou 'Retour', reçu : '{direction}'")

    arrets   = ARRETS_ALLER   if direction == "Aller" else ARRETS_RETOUR
    dist_tot = DIST_TOT_ALLER if direction == "Aller" else DIST_TOT_RETOUR
    dir_enc  = 0              if direction == "Aller" else 1

    # Contexte temporel
    now   = datetime.now()
    h_dec = pos.heure_decimal if pos.heure_decimal is not None \
            else now.hour + now.minute / 60 + now.second / 3600
    jour  = pos.jour_semaine  if pos.jour_semaine  is not None \
            else now.weekday()

    # ── CORRECTION WEEK-END ALGÉRIEN : vendredi(4) + samedi(5) ──
    est_we = int(jour in [4, 5])

    h_sin   = math.sin(2 * math.pi * h_dec / 24)
    h_cos   = math.cos(2 * math.pi * h_dec / 24)
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

        # Vecteur de features — MÊME ORDRE que l'entraînement XGBoost
        X = np.array([[
            h_dec, h_sin, h_cos, jour, est_we,
            pos.latitude, pos.longitude, pos.altitude_m, pos.cap_deg,
            pos.vitesse_kmh, pos.vitesse_moy_60s,
            pos.acceleration_kmh_s, pos.pente_pct,
            pct_bus, pos.temps_arret_cumule_s, dir_enc,
            dv,                               # dist_vers_arret_m
            arret["dist"] / dist_tot * 100,   # pct_vers_arret
            math.sqrt(dv),                    # dist_vers_arret_sqrt
            dv / dist_tot,                    # ratio_dist
        ]])

        # ── Prédictions XGBoost (3 modèles) ──────────────────────
        eta_s  = float(max(0.0, MODEL_ETA.predict(X)[0]))
        bas_s  = float(max(0.0, MODEL_BAS.predict(X)[0]))
        haut_s = float(max(bas_s, MODEL_HAUT.predict(X)[0]))

        # ── Appliquer le coefficient trafic ──────────────────────
        eta_s_c  = eta_s  * coeff
        bas_s_c  = bas_s  * coeff
        haut_s_c = haut_s * coeff

        # ── Indice de confiance ───────────────────────────────────
        confiance = get_confiance(eta_s_c, bas_s_c, haut_s_c)

        resultats.append(ArretETA(
            nom          = arret["nom"],
            lat          = arret["lat"],
            lon          = arret["lon"],
            eta_min      = round(eta_s_c  / 60, 1),
            eta_s        = int(eta_s_c),
            dist_km      = round(dv / 1000, 2),
            eta_min_bas  = round(bas_s_c  / 60, 1),
            eta_min_haut = round(haut_s_c / 60, 1),
            confiance    = confiance,
        ))

    if pos.n_arrets_max:
        resultats = resultats[:pos.n_arrets_max]

    return resultats, coeff, nb_bus_devant


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service"        : "DZ Bus Tracker — API ETA v5.0",
        "version"        : "5.0",
        "algorithme"     : "XGBoost",
        "modele_version" : "5.0-xgboost",
        "date_modele"    : "2026-06-17",
        "mae_global_min" : 1.86,
        "mae_aller_min"  : 1.07,
        "mae_retour_min" : 2.32,
        "r2_global"      : 0.8992,
        "firebase_trafic": _firebase_ok,
        "endpoints": {
            "POST /predict"           : "ETA par arrêt avec intervalles de confiance",
            "GET  /health"            : "Santé du serveur",
            "GET  /arrets/{direction}": "Liste des arrêts Aller ou Retour",
        }
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
    """Retourne la liste des arrêts d'une direction."""
    if direction == "Aller":
        return {"direction": "Aller", "arrets": ARRETS_ALLER}
    elif direction == "Retour":
        return {"direction": "Retour", "arrets": ARRETS_RETOUR}
    else:
        raise HTTPException(
            status_code=400,
            detail="direction doit être 'Aller' ou 'Retour'"
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(pos: BusPosition):
    """
    Endpoint principal — appelé par passager.tsx à chaque tap sur un bus.

    Algorithme XGBoost v5.0 :
    - 3 modèles : eta (central), bas (P10), haut (P90)
    - Intervalle de confiance via régression quantile
    - Correction trafic via intelligence collective des bus actifs

    Exemple de requête :
    {
        "latitude": 36.755,
        "longitude": 5.060,
        "vitesse_kmh": 28.5,
        "distance_parcourue_m": 2550,
        "direction": "Aller",
        "temps_arret_cumule_s": 45,
        "firebase_bus_actifs": {
            "Bus_ABC": {
                "direction": "Aller",
                "latitude": 36.76,
                "longitude": 5.07,
                "vitesse": 12
            }
        }
    }

    Exemple de réponse :
    {
        "arrets": [
            {
                "nom": "Arrêt 8",
                "eta_min": 1.4,
                "eta_s": 84,
                "dist_km": 0.36,
                "eta_min_bas": 1.1,
                "eta_min_haut": 1.8,
                "confiance": "haute"
            },
            ...
        ],
        "coefficient_trafic": 1.0,
        "condition_trafic": "fluide"
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
        direction          = pos.direction,
        nb_arrets          = len(arrets),
        arrets             = arrets,
        bus_position       = {
            "latitude":             pos.latitude,
            "longitude":            pos.longitude,
            "vitesse_kmh":          pos.vitesse_kmh,
            "distance_parcourue_m": pos.distance_parcourue_m,
        },
        modele_date        = PKG["date"],
        timestamp          = datetime.now().isoformat(),
        coefficient_trafic = round(coeff, 2),
        nb_bus_devant      = nb_bus_devant,
        condition_trafic   = condition_trafic,
    )
