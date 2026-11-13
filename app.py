# -*- coding: utf-8 -*-
"""
CyberVion Analytics - Détection et prédiction de 10 types d'attaques réseau.

Application Streamlit basée sur la logique du notebook
"Détection multi-classes des cyberattaques avec Machine Learning" :
  chargement du CSV -> doublons -> NaN -> encodage -> train/val/test (70/15/15)
  -> standardisation -> 5 modèles -> comparaison -> test final -> prédiction.

Lancement :  streamlit run app.py
Identifiants : voir USERNAME / PASSWORD ci-dessous.
"""

import base64
import hmac
import os
import time
import warnings
from pathlib import Path
from string import Template

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
APP_NAME = "CyberVion Analytics"
USERNAME = "koukita"
PASSWORD = "KAU@S1029#"

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset.csv"
RANDOM_STATE = 42
TARGET = "attack_type"
DROP_COLS = ["attack_type", "is_attack", "sample_id", "timestamp_utc",
             "src_ip", "dst_ip", "src_port", "service"]
MODELS_WITH_PROBA = ["Régression logistique", "Random Forest", "XGBoost"]
NO_SCALING = ["Random Forest", "XGBoost"]
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONTHS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
             "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def make_favicon():
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((2, 2, 62, 62), 14, fill=(37, 99, 235, 255))
        d.polygon([(32, 11), (50, 18), (50, 32), (32, 53), (14, 32), (14, 18)],
                  fill=(255, 255, 255, 255))
        d.ellipse((24, 24, 40, 40), fill=(37, 99, 235, 255))
        d.ellipse((29, 29, 35, 35), fill=(255, 255, 255, 255))
        return im
    except Exception:
        return None


st.set_page_config(page_title=APP_NAME, page_icon=make_favicon(),
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# INFORMATIONS SUR LES CLASSES
# ----------------------------------------------------------------------------
CLASS_COLORS = {
    "DDoS": "#4f46e5", "Phishing": "#8b5cf6", "SQL_Injection": "#3b82f6",
    "Brute_Force": "#22d3ee", "XSS": "#14b8a6", "Malware": "#f59e0b",
    "Ransomware": "#ef4444", "Port_Scanning": "#ec4899",
    "Man_in_the_Middle": "#84cc16", "Botnet": "#64748b",
}
CLASS_INFO = {
    "DDoS": ("Déni de service distribué",
             "Saturation du service cible par un volume anormal de paquets et de requêtes, ce qui provoque une indisponibilité.",
             "Activer la limitation de débit, filtrer en amont et basculer vers une protection anti-DDoS."),
    "Phishing": ("Hameçonnage",
                 "Messages contenant des liens frauduleux, souvent émis depuis des domaines récents ou échouant au contrôle SPF.",
                 "Bloquer le domaine expéditeur, signaler les liens et sensibiliser les utilisateurs."),
    "SQL_Injection": ("Injection SQL",
                      "Requêtes comportant des mots-clés SQL anormaux visant à manipuler ou extraire le contenu de la base de données.",
                      "Utiliser des requêtes paramétrées, activer un pare-feu applicatif et auditer les journaux."),
    "Brute_Force": ("Force brute",
                    "Tentatives d'authentification échouées et répétées sur un service d'accès (SSH, RDP, FTP).",
                    "Verrouiller temporairement le compte, activer l'authentification multifacteur et filtrer la source."),
    "XSS": ("Cross-Site Scripting",
            "Injection de scripts dans des pages web, détectée par des motifs de balises ou d'événements suspects.",
            "Encoder les sorties, appliquer une politique CSP et valider les entrées utilisateur."),
    "Malware": ("Logiciel malveillant",
                "Création de processus et d'exécutables suspects accompagnée d'une activité réseau sortante inhabituelle.",
                "Isoler l'hôte, lancer une analyse EDR et révoquer les identifiants potentiellement exposés."),
    "Ransomware": ("Rançongiciel",
                   "Modifications massives de fichiers avec une forte entropie, signe typique d'un chiffrement de données.",
                   "Isoler immédiatement la machine, couper les partages réseau et restaurer depuis des sauvegardes saines."),
    "Port_Scanning": ("Balayage de ports",
                      "Sondage d'un grand nombre de ports distincts sur une courte durée pour cartographier les services exposés.",
                      "Bloquer la source, durcir le pare-feu et fermer les ports inutiles."),
    "Man_in_the_Middle": ("Homme du milieu",
                          "Interception du trafic par usurpation ARP et erreurs de certificats TLS.",
                          "Activer l'inspection ARP dynamique, imposer TLS/HSTS et vérifier les certificats."),
    "Botnet": ("Réseau de machines zombies",
               "Communications régulières de type balise vers un serveur de commande et de contrôle, avec de nombreuses requêtes DNS.",
               "Bloquer les domaines de commande, isoler les hôtes infectés et analyser le trafic DNS."),
}


def disp(c):
    return {"Man_in_the_Middle": "Man-in-the-Middle"}.get(c, c.replace("_", " "))


def ccolor(c):
    return CLASS_COLORS.get(c, "#64748b")


# Compteurs de vitesse (paramètres modifiables du trafic)
GAUGE_SPECS = {
    "packet_rate_pps": dict(label="Taux de paquets", unit="pps", min=0, max=20000, step=10, power=3, decimals=0),
    "flow_duration_ms": dict(label="Durée du flux", unit="ms", min=0, max=120000, step=50, power=3, decimals=0),
    "avg_packet_size_bytes": dict(label="Taille moyenne du paquet", unit="octets", min=0, max=1500, step=1, power=1, decimals=0),
    "payload_entropy": dict(label="Entropie de la charge utile", unit="bits", min=0, max=8, step=0.01, power=1, decimals=2),
    "unique_dst_ports_5m": dict(label="Ports distincts (5 min)", unit="ports", min=0, max=400, step=1, power=2, decimals=0),
    "failed_logins_5m": dict(label="Tentatives échouées", unit="essais", min=0, max=120, step=1, power=2, decimals=0),
    "http_requests_5m": dict(label="Requêtes HTTP (5 min)", unit="req", min=0, max=150, step=1, power=2, decimals=0),
    "sql_keyword_hits_5m": dict(label="Mots-clés SQL détectés", unit="hits", min=0, max=20, step=1, power=1, decimals=0),
    "xss_pattern_hits_5m": dict(label="Motifs XSS détectés", unit="hits", min=0, max=20, step=1, power=1, decimals=0),
    "arp_spoof_events_5m": dict(label="Événements ARP suspects", unit="événements", min=0, max=35, step=1, power=1, decimals=0),
    "tls_cert_errors_5m": dict(label="Erreurs de certificat TLS", unit="erreurs", min=0, max=22, step=1, power=1, decimals=0),
    "email_links_count": dict(label="Liens dans l'e-mail", unit="liens", min=0, max=12, step=1, power=1, decimals=0),
    "file_modifications_5m": dict(label="Modifications de fichiers", unit="fichiers", min=0, max=2000, step=5, power=3, decimals=0),
    "process_spawns_5m": dict(label="Processus créés", unit="proc.", min=0, max=70, step=1, power=2, decimals=0),
    "dns_queries_5m": dict(label="Requêtes DNS (5 min)", unit="req", min=0, max=150, step=1, power=2, decimals=0),
    "outbound_connections_5m": dict(label="Connexions sortantes", unit="conn.", min=0, max=190, step=1, power=2, decimals=0),
}
GAUGES_FULL = ["packet_rate_pps", "flow_duration_ms", "avg_packet_size_bytes", "payload_entropy",
               "unique_dst_ports_5m", "failed_logins_5m", "http_requests_5m", "sql_keyword_hits_5m",
               "xss_pattern_hits_5m", "arp_spoof_events_5m", "tls_cert_errors_5m", "email_links_count",
               "file_modifications_5m", "process_spawns_5m", "dns_queries_5m", "outbound_connections_5m"]
GAUGES_COMPACT = ["packet_rate_pps", "flow_duration_ms", "unique_dst_ports_5m",
                  "failed_logins_5m", "payload_entropy", "sql_keyword_hits_5m"]
GLOBAL_SCENARIO = "Profil moyen du trafic"

# ----------------------------------------------------------------------------
# THÈMES (clair = image 1, sombre = image 2)
# ----------------------------------------------------------------------------
DARK = dict(
    name="dark", bg="#040b1f", bg2="#0a1c4d", card="rgba(13,31,77,0.62)", cell="rgba(8,22,60,0.75)",
    border="rgba(96,150,255,0.24)", text="#e8efff", muted="#8ea6d8", accent="#3b82f6",
    accent2="#22d3ee", track="rgba(120,160,255,0.20)", tick="rgba(150,180,255,0.35)",
    grid="rgba(120,160,255,0.12)", inp="rgba(6,18,52,0.9)", danger="#ef4444", ok="#22c55e",
    warn="#f59e0b", side="linear-gradient(180deg,#071536 0%,#040b1f 100%)",
    shadow="0 8px 30px rgba(0,0,0,0.35)", needle="#e8efff", btn="linear-gradient(90deg,#2563eb,#06b6d4)",
    row="rgba(120,160,255,0.07)", glow="rgba(59,130,246,0.35)",
)
LIGHT = dict(
    name="light", bg="#eef3fb", bg2="#e6eefb", card="#ffffff", cell="#f5f8fe",
    border="#dbe5f5", text="#10234f", muted="#5f7299", accent="#4f5bd5",
    accent2="#2563eb", track="#e3eaf7", tick="#c3cfe6", grid="#e6edf8", inp="#ffffff",
    danger="#e11d48", ok="#16a34a", warn="#d97706",
    side="linear-gradient(180deg,#ffffff 0%,#f4f8ff 100%)",
    shadow="0 4px 18px rgba(40,70,140,0.08)", needle="#1c2b5a", btn="linear-gradient(90deg,#4f46e5,#2563eb)",
    row="#f3f7fd", glow="rgba(79,91,213,0.25)",
)


def theme():
    return DARK if st.session_state.get("dark_mode", True) else LIGHT


NAV_ICONS = {
    "Tableau de bord": "<rect x='3' y='3' width='8' height='8' rx='2'/><rect x='13' y='3' width='8' height='8' rx='2'/><rect x='3' y='13' width='8' height='8' rx='2'/><rect x='13' y='13' width='8' height='8' rx='2'/>",
    "Prédiction / Test": "<path d='M9 3h6M10 3v6L4.5 19a2 2 0 001.8 3h11.4a2 2 0 001.8-3L14 9V3'/>",
    "Analyses": "<path d='M4 20V10M10 20V4M16 20v-7M22 20H2'/>",
    "Incidents": "<path d='M12 3l10 18H2z'/><path d='M12 10v5M12 18v.5'/>",
    "Données & Modèle": "<ellipse cx='12' cy='6' rx='8' ry='3'/><path d='M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6'/>",
    "Évaluation": "<circle cx='12' cy='12' r='9'/><path d='M8 12.5l3 3 5-6'/>",
    "Paramètres": "<path d='M4 7h10M18 7h2M4 17h2M10 17h10'/><circle cx='16' cy='7' r='2'/><circle cx='8' cy='17' r='2'/>",
}
PAGES = list(NAV_ICONS.keys())


def nav_css():
    out = []
    for i, name in enumerate(PAGES, start=1):
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
               "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>" + NAV_ICONS[name] + "</svg>")
        out.append("section[data-testid='stSidebar'] [data-testid='stRadioGroup'] > div:nth-of-type(%d) [data-testid='stRadioOption']::before"
                   "{-webkit-mask-image:url(\"data:image/svg+xml;utf8,%s\");mask-image:url(\"data:image/svg+xml;utf8,%s\");}"
                   % (i, svg, svg))
    return "\n".join(out)


CSS = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family: ${font}; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, ${bg2} 0%, ${bg} 60%); color: ${text}; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu { display: none !important; }
.block-container { padding: 1.1rem 1.6rem 2rem 1.6rem; max-width: 100%; }
h1, h2, h3, h4, h5 { color: ${text}; font-family: ${font}; }
.stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] { color: ${text}; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label p { color: ${muted} !important; font-size: 12px; font-weight: 600; letter-spacing: .2px; }
[data-testid="stVerticalBlock"] { gap: 0.9rem; }

/* ---------- Barre latérale ---------- */
section[data-testid="stSidebar"] { background: ${side}; border-right: 1px solid ${border}; width: 260px !important; }
section[data-testid="stSidebar"] > div { padding-top: .6rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none; }
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 6px; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  position: relative; padding: 11px 14px 11px 46px; border-radius: 12px; cursor: pointer;
  border: 1px solid transparent; width: 100%; transition: all .15s ease; margin: 0; box-sizing: border-box; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p { color: ${muted} !important; font-weight: 600; font-size: 14px; margin: 0; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]::before {
  content: ""; position: absolute; left: 15px; top: 50%; width: 19px; height: 19px; margin-top: -9.5px;
  background-color: ${muted}; -webkit-mask-size: contain; mask-size: contain;
  -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: ${row}; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: ${btn}; box-shadow: 0 6px 18px ${glow}; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color: #fff !important; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before { background-color: #fff; }
${navicons}

.brand { display: flex; align-items: center; gap: 11px; padding: 6px 6px 14px 6px; border-bottom: 1px solid ${border}; margin-bottom: 14px; }
.brand svg { width: 46px; height: 46px; flex: none; filter: drop-shadow(0 4px 10px ${glow}); }
.brand-t { font-weight: 800; font-size: 19px; line-height: 1.05; color: ${text}; letter-spacing: -.3px; }
.brand-t b { color: ${accent2}; font-weight: 800; }
.brand-s { font-size: 9.5px; letter-spacing: 3.2px; color: ${muted}; font-weight: 700; margin-top: 3px; }
.side-card { margin-top: 18px; padding: 16px 12px; border-radius: 16px; text-align: center; border: 1px solid ${border}; background: ${card}; }
.side-card svg { width: 54px; height: 54px; color: ${accent2}; filter: drop-shadow(0 0 12px ${glow}); }
.side-card .t { font-weight: 800; font-size: 14px; margin-top: 4px; color: ${text}; }
.side-card .s { font-size: 10.5px; color: ${muted}; margin-top: 3px; line-height: 1.4; }

/* ---------- En-tête ---------- */
.cv-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 20px; border-radius: 18px;
  background: ${card}; border: 1px solid ${border}; box-shadow: ${shadow}; margin-bottom: 4px; }
.cv-head-l { display: flex; align-items: center; gap: 16px; }
.cv-shield { width: 50px; height: 50px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
  background: ${btn}; color: #fff; box-shadow: 0 6px 18px ${glow}; flex: none; }
.cv-shield svg { width: 28px; height: 28px; }
.cv-title { font-size: 23px; font-weight: 800; color: ${text}; letter-spacing: -.4px; line-height: 1.15; }
.cv-sub { font-size: 12.5px; color: ${muted}; margin-top: 3px; font-weight: 500; }
.cv-sub i { font-style: normal; margin: 0 7px; color: ${accent2}; }
.cv-user { display: flex; align-items: center; gap: 10px; padding: 6px 14px 6px 6px; border-radius: 999px; border: 1px solid ${border}; background: ${cell}; }
.cv-av { width: 32px; height: 32px; border-radius: 50%; background: ${btn}; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }
.cv-un { font-size: 13px; font-weight: 700; color: ${text}; line-height: 1.1; }
.cv-ur { font-size: 10.5px; color: ${muted}; }
.pg-title { font-size: 19px; font-weight: 800; color: ${text}; margin: 6px 2px 0 2px; }
.pg-sub { font-size: 12.5px; color: ${muted}; margin: -6px 2px 0 2px; }

/* ---------- Cartes ---------- */
[class*="st-key-card_"] { background: ${card}; border: 1px solid ${border}; border-radius: 18px; padding: 16px 18px 12px 18px;
  box-shadow: ${shadow}; backdrop-filter: blur(6px); }
.ch { display: flex; align-items: center; gap: 9px; font-weight: 700; font-size: 14.5px; color: ${text}; margin-bottom: 2px; }
.ch svg { width: 17px; height: 17px; color: ${accent2}; flex: none; }
.ch small { margin-left: auto; font-weight: 500; color: ${muted}; font-size: 11.5px; }

.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 1350px) { .kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 800px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { display: flex; align-items: center; gap: 13px; padding: 15px 16px; border-radius: 16px; background: ${card}; border: 1px solid ${border}; box-shadow: ${shadow}; }
.kpi-i { width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none;
  background: ${cell}; border: 1px solid ${border}; color: ${accent2}; }
.kpi-i svg { width: 22px; height: 22px; }
.kpi-l { font-size: 11.5px; color: ${muted}; font-weight: 600; }
.kpi-v { font-size: 25px; font-weight: 800; color: ${text}; letter-spacing: -.5px; line-height: 1.15; }
.kpi-s { font-size: 10.8px; color: ${ok}; font-weight: 600; }

/* ---------- Tableaux ---------- */
.tbl-w { margin-bottom: 12px; overflow-x: auto; max-height: 520px; overflow-y: auto; border-radius: 12px; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.tbl th, .tbl td { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
.tbl th { text-align: left; padding: 9px 10px; font-size: 11px; color: ${muted}; font-weight: 700; border-bottom: 1px solid ${border}; position: sticky; top: 0; background: ${cardsolid}; }
.tbl td { padding: 9px 10px; color: ${text}; border-bottom: 1px solid ${grid}; white-space: nowrap; }
.tbl tr:hover td { background: ${row}; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 7px; font-size: 11px; font-weight: 700; background: rgba(239,68,68,.16); color: ${danger}; border: 1px solid rgba(239,68,68,.35); }
.badge.ok { background: rgba(34,197,94,.14); color: ${ok}; border-color: rgba(34,197,94,.35); }
.dotc { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }
.chip { display: inline-block; padding: 4px 13px; border-radius: 8px; font-weight: 700; font-size: 12.5px; color: #fff; }

/* ---------- Résultat de prédiction ---------- */
.res { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 16px 18px; border-radius: 16px;
  background: linear-gradient(135deg, rgba(239,68,68,.16), rgba(239,68,68,.05)); border: 1px solid rgba(239,68,68,.40); }
.res-l { display: flex; align-items: center; gap: 14px; }
.res-ring { width: 54px; height: 54px; border-radius: 50%; border: 2px solid ${danger}; display: flex; align-items: center; justify-content: center; color: ${danger}; flex: none; }
.res-ring svg { width: 28px; height: 28px; }
.res-k { font-size: 11px; color: ${muted}; font-weight: 600; }
.res-v { font-size: 28px; font-weight: 800; color: ${danger}; line-height: 1.1; letter-spacing: -.4px; }
.res-r { min-width: 150px; }
.res-c { font-size: 28px; font-weight: 800; color: ${text}; line-height: 1.1; }
.res-bar { height: 6px; border-radius: 6px; background: ${track}; margin-top: 7px; overflow: hidden; }
.res-bar i { display: block; height: 100%; border-radius: 6px; background: ${danger}; }
.res-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-top: 12px; font-size: 12px; color: ${muted}; }
.res-meta b { color: ${text}; }
.top3 { margin-top: 10px; display: grid; gap: 7px; }
.top3 .r { display: grid; grid-template-columns: 120px 1fr 62px; align-items: center; gap: 10px; font-size: 12px; color: ${text}; }
.top3 .b { height: 7px; border-radius: 7px; background: ${track}; overflow: hidden; }
.top3 .b i { display: block; height: 100%; border-radius: 7px; }
.top3 .p { text-align: right; font-weight: 700; }
.note { font-size: 12.5px; color: ${muted}; line-height: 1.55; margin-top: 12px; }
.note b { color: ${text}; }
.empty { padding: 28px 16px; text-align: center; color: ${muted}; font-size: 13px; border: 1px dashed ${border}; border-radius: 14px; }

/* ---------- Widgets ---------- */
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button { border-radius: 12px; font-weight: 700; border: 1px solid ${border} !important; background: ${cell} !important; color: ${text} !important; padding: .55rem 1rem; transition: all .15s ease; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover { border-color: ${accent2} !important; color: ${accent2} !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { border-radius: 12px; font-weight: 700; background: ${btn} !important; color: #fff !important; border: none !important; box-shadow: 0 8px 22px ${glow}; padding: .6rem 1rem; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.12); }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p, [data-testid="stBaseButton-secondary"] p, [data-testid="stDownloadButton"] p { color: inherit !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stTextInput"] [data-baseweb], [data-testid="stTextInput"] [role="group"] {
  background: ${inp} !important; border: 1px solid ${border} !important; border-radius: 11px !important; }
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-testid="stTextInput"] input { color: ${text} !important; -webkit-text-fill-color: ${text} !important; background: transparent !important; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stMultiSelect"] input::placeholder { color: ${muted} !important; -webkit-text-fill-color: ${muted} !important; opacity: .8; }
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg, [data-testid="stTextInput"] svg { color: ${muted} !important; fill: ${muted} !important; }
[data-testid="stTextInput"] button { background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] > div, [data-testid="stMultiSelectVirtualDropdown"] > div {
  background: ${cardsolid} !important; border-radius: 12px; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectVirtualDropdown"] { border: 1px solid ${border}; box-shadow: ${shadow}; }
[role="listbox"] [role="option"] { background: transparent !important; color: ${text} !important; }
[role="listbox"] [role="option"] * { color: ${text} !important; }
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][data-focused="true"], [role="listbox"] [role="option"][aria-selected="true"] { background: ${row} !important; }
[data-testid="stCheckbox"] label > div:first-of-type { background-color: ${track} !important; }
[data-testid="stCheckbox"][data-selected="true"] label > div:first-of-type { background-color: ${accent} !important; }
[data-testid="stTabs"] button[role="tab"], button[data-baseweb="tab"] { color: ${muted} !important; font-weight: 700; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] { color: ${accent2} !important; }
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid ${border}; }
[data-baseweb="tab-highlight"] { background: ${accent2} !important; }
[data-baseweb="tab-border"] { background: ${border} !important; }
[data-testid="stFileUploaderDropzone"] { background: ${cell}; border: 1px dashed ${border}; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] * { color: ${muted} !important; }
[data-testid="stFileUploaderDropzone"] button { color: ${text} !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: ${muted} !important; }
[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stCheckbox"] label p { color: ${text} !important; font-size: 13px; }
iframe { border: none; }

/* ---------- Connexion ---------- */
.login-hero { text-align: center; margin: 7vh 0 18px 0; }
.login-hero svg { width: 84px; height: 84px; filter: drop-shadow(0 8px 24px ${glow}); }
.login-hero .n { font-size: 32px; font-weight: 800; color: ${text}; letter-spacing: -.8px; margin-top: 8px; }
.login-hero .n b { color: ${accent2}; }
.login-hero .a { font-size: 11px; letter-spacing: 5px; color: ${muted}; font-weight: 700; margin-top: 2px; }
.login-hero .d { font-size: 14px; color: ${muted}; margin-top: 12px; }
</style>
""")

LOGO_SVG = ("<svg viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><defs>"
            "<linearGradient id='cvg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#22d3ee'/>"
            "<stop offset='1' stop-color='#4f46e5'/></linearGradient></defs>"
            "<path d='M32 3l23 8.5v17C55 43.5 45 54 32 61 19 54 9 43.500 9 28.500v-17z' fill='url(#cvg)'/>"
            "<path d='M32 8.500l18 6.700v13.300c0 11.500-7.500 20-18 26-10.500-6-18-14.500-18-26V15.200z' fill='none' stroke='rgba(255,255,255,.35)' stroke-width='1.200'/>"
            "<ellipse cx='32' cy='28' rx='14' ry='8.500' fill='none' stroke='#fff' stroke-width='2.600'/>"
            "<circle cx='32' cy='28' r='5.400' fill='#fff'/><circle cx='32' cy='28' r='2.300' fill='#0b1b4d'/>"
            "<path d='M22 43l6-4 4 3.500 4-3.500 6 4' fill='none' stroke='#fff' stroke-width='1.800' stroke-linecap='round' stroke-linejoin='round'/>"
            "<circle cx='22' cy='43' r='1.900' fill='#fff'/><circle cx='28' cy='39' r='1.900' fill='#fff'/>"
            "<circle cx='36' cy='39' r='1.900' fill='#fff'/><circle cx='42' cy='43' r='1.900' fill='#fff'/></svg>")


def ico(name, size=None):
    paths = {
        "flux": "<path d='M3 12h4l3-8 4 16 3-8h4'/>",
        "shield": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/>",
        "shieldcheck": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/><path d='M8.5 12l2.5 2.5 4.5-5'/>",
        "shieldx": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/><path d='M12 8.5v4.5M12 16v.5'/>",
        "target": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='4.5'/><circle cx='12' cy='12' r='.8'/>",
        "db": "<ellipse cx='12' cy='6' rx='8' ry='3'/><path d='M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6'/>",
        "layers": "<path d='M12 3l9 5-9 5-9-5z'/><path d='M3 13l9 5 9-5'/>",
        "alert": "<path d='M12 3l10 18H2z'/><path d='M12 10v5M12 18v.5'/>",
        "flask": "<path d='M9 3h6M10 3v6L4.5 19a2 2 0 001.8 3h11.4a2 2 0 001.8-3L14 9V3'/>",
        "clock": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
        "chart": "<path d='M4 20V10M10 20V4M16 20v-7M22 20H2'/>",
        "pie": "<path d='M12 3a9 9 0 109 9h-9z'/><path d='M15 3.5A9 9 0 0120.5 9H15z'/>",
        "list": "<path d='M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01'/>",
        "gauge": "<path d='M4 17a9 9 0 1116 0'/><path d='M12 17l4-6'/>",
        "grid": "<rect x='3' y='3' width='8' height='8' rx='2'/><rect x='13' y='3' width='8' height='8' rx='2'/><rect x='3' y='13' width='8' height='8' rx='2'/><rect x='13' y='13' width='8' height='8' rx='2'/>",
    }
    s = f" width='{size}' height='{size}'" if size else ""
    return (f"<svg viewBox='0 0 24 24'{s} fill='none' stroke='currentColor' stroke-width='1.8' "
            f"stroke-linecap='round' stroke-linejoin='round'>{paths[name]}</svg>")


def html(s):
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def inject_css():
    T = dict(theme())
    T["font"] = FONT
    T["navicons"] = nav_css()
    T["cardsolid"] = "#0b1b46" if T["name"] == "dark" else "#ffffff"
    st.markdown(CSS.safe_substitute(T), unsafe_allow_html=True)


def card_title(icon, title, small=""):
    sm = f"<small>{small}</small>" if small else ""
    html(f"<div class='ch'>{ico(icon)}<span>{title}</span>{sm}</div>")


def fr(x, d=2):
    return f"{x:,.{d}f}".replace(",", "\u202f").replace(".", ",")


def pct(x, d=2):
    return fr(100 * x, d) + "\u00a0%"


def style_fig(fig, h=260, legend=True):
    T = theme()
    fig.update_layout(
        height=h, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"], family=FONT, size=11),
        showlegend=legend, legend=dict(font=dict(color=T["text"], size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["grid"], tickfont=dict(color=T["muted"]), automargin=True)
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["grid"], tickfont=dict(color=T["muted"]), automargin=True)
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False})


def html_table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tbl-w'><table class='tbl'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


# ----------------------------------------------------------------------------
# COMPOSANT : COMPTEURS DE VITESSE INTERACTIFS
# ----------------------------------------------------------------------------
GAUGE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
html,body{margin:0;padding:0;background:transparent;overflow:hidden;font-family:Inter,'Segoe UI',system-ui,sans-serif}
#grid{display:grid;gap:10px;padding:2px}
.g{border-radius:16px;padding:6px 4px 9px;text-align:center;user-select:none;-webkit-user-select:none;touch-action:none;cursor:grab;outline:none;transition:box-shadow .15s}
.g:active{cursor:grabbing}
.g:focus-visible{box-shadow:0 0 0 2px var(--acc)}
.g svg{width:100%;height:auto;display:block}
.nm{font-size:12px;font-weight:700;margin-top:-8px;line-height:1.25;padding:0 6px}
</style></head>
<body><div id="grid"></div>
<script>
const SW=270,A0=135,CX=100,CY=86,R=70;
let args=null,state={},token=null,dragging=null,lastSent="",timer=null;
const grid=document.getElementById("grid");
function send(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data||{}),"*");}
function polar(a,r){const t=a*Math.PI/180;return [CX+r*Math.cos(t),CY+r*Math.sin(t)];}
function arc(a1,a2,r){const p1=polar(a1,r),p2=polar(a2,r);const large=(a2-a1)>180?1:0;
  return "M"+p1[0].toFixed(2)+" "+p1[1].toFixed(2)+" A"+r+" "+r+" 0 "+large+" 1 "+p2[0].toFixed(2)+" "+p2[1].toFixed(2);}
function tOf(g,v){const f=(v-g.min)/(g.max-g.min);return Math.pow(Math.min(1,Math.max(0,f)),1/g.power);}
function vOf(g,t){let v=g.min+(g.max-g.min)*Math.pow(t,g.power);v=Math.round(v/g.step)*g.step;
  v=Math.min(g.max,Math.max(g.min,v));return +v.toFixed(Math.max(g.decimals,2));}
function fmt(g,v){return v.toLocaleString("fr-FR",{minimumFractionDigits:g.decimals,maximumFractionDigits:g.decimals});}
function short(v){if(v>=1000){return (v/1000).toFixed(v>=10000?0:1).replace(".0","")+"k";}return String(+v.toFixed(1)).replace(".",",");}

function build(){
  const T=args.theme;
  document.documentElement.style.setProperty("--acc",T.accent2);
  grid.style.gridTemplateColumns="repeat("+args.cols+",minmax(0,1fr))";
  grid.innerHTML="";
  args.gauges.forEach(function(g,i){
    const d=document.createElement("div");
    d.className="g";d.tabIndex=0;d.dataset.i=i;
    d.style.background=T.cell;d.style.border="1px solid "+T.border;
    let ticks="",labels="";
    for(let k=0;k<=20;k++){
      const a=A0+SW*k/20,major=(k%5===0);
      const p1=polar(a,major?50:54),p2=polar(a,60);
      ticks+='<line data-k="'+k+'" x1="'+p1[0].toFixed(2)+'" y1="'+p1[1].toFixed(2)+'" x2="'+p2[0].toFixed(2)+'" y2="'+p2[1].toFixed(2)+'" stroke-width="'+(major?2:1.2)+'" stroke-linecap="round"/>';
      if(major&&k>0&&k<20){const q=polar(a,40);const gv=vOf(g,k/20);
        labels+='<text x="'+q[0].toFixed(2)+'" y="'+(q[1]+2.6).toFixed(2)+'" text-anchor="middle" font-size="7.5" font-weight="600" fill="'+T.muted+'">'+short(gv)+'</text>';}
    }
    d.innerHTML='<svg viewBox="0 0 200 150"><defs><linearGradient id="gr'+i+'" x1="0" y1="1" x2="1" y2="0">'
      +'<stop offset="0" stop-color="'+T.accent2+'"/><stop offset="1" stop-color="'+T.accent+'"/></linearGradient></defs>'
      +'<path d="'+arc(A0,A0+SW,R)+'" fill="none" stroke="'+T.track+'" stroke-width="9" stroke-linecap="round"/>'
      +'<path class="val" fill="none" stroke="url(#gr'+i+')" stroke-width="9" stroke-linecap="round"/>'
      +'<g class="tk">'+ticks+'</g><g>'+labels+'</g>'
      +'<text x="46" y="143" text-anchor="middle" font-size="8" font-weight="600" fill="'+T.muted+'">'+short(g.min)+'</text>'
      +'<text x="154" y="143" text-anchor="middle" font-size="8" font-weight="600" fill="'+T.muted+'">'+short(g.max)+'</text>'
      +'<polygon class="nd" fill="'+T.needle+'"/>'
      +'<circle cx="100" cy="86" r="7.5" fill="'+T.cell+'" stroke="'+T.accent+'" stroke-width="3"/>'
      +'<text class="num" x="100" y="122" text-anchor="middle" font-size="20" font-weight="800" fill="'+T.text+'"></text>'
      +'<text class="un" x="100" y="134" text-anchor="middle" font-size="8.5" font-weight="600" fill="'+T.muted+'"></text></svg>'
      +'<div class="nm" style="color:'+T.text+'">'+g.label+'</div>';
    grid.appendChild(d);
    update(i);
  });
  fit();
}
function update(i){
  const g=args.gauges[i],T=args.theme,d=grid.children[i];
  const v=state[g.key],t=tOf(g,v);
  d.querySelector(".val").setAttribute("d",arc(A0,A0+Math.max(0.6,SW*t),R));
  const ang=A0+SW*t,tip=polar(ang,52),b1=[CX+3.4*Math.cos((ang+90)*Math.PI/180),CY+3.4*Math.sin((ang+90)*Math.PI/180)],
        b2=[CX+3.4*Math.cos((ang-90)*Math.PI/180),CY+3.4*Math.sin((ang-90)*Math.PI/180)],tail=polar(ang+180,10);
  d.querySelector(".nd").setAttribute("points",[tip,b1,tail,b2].map(function(p){return p[0].toFixed(2)+","+p[1].toFixed(2);}).join(" "));
  d.querySelectorAll(".tk line").forEach(function(l){l.setAttribute("stroke",(+l.dataset.k/20)<=t+0.001?T.accent2:T.tick);});
  d.querySelector(".num").textContent=fmt(g,v);
  d.querySelector(".un").textContent=g.unit;
}
function fit(){send("streamlit:setFrameHeight",{height:grid.offsetHeight+6});}
function commit(){const s=JSON.stringify(state);if(s!==lastSent){lastSent=s;send("streamlit:setComponentValue",{value:JSON.parse(s),dataType:"json"});}}
function move(e){
  const d=grid.children[dragging],svg=d.querySelector("svg"),r=svg.getBoundingClientRect();
  const cx=r.left+r.width*CX/200,cy=r.top+r.height*CY/150,dx=e.clientX-cx,dy=e.clientY-cy;
  if(Math.sqrt(dx*dx+dy*dy)<r.width*0.05)return;
  let a=Math.atan2(dy,dx)*180/Math.PI;if(a<0)a+=360;
  let rel=a-A0;if(rel<0)rel+=360;
  if(rel>SW){rel=(rel>SW+(360-SW)/2)?0:SW;}
  const g=args.gauges[dragging];state[g.key]=vOf(g,rel/SW);update(dragging);
}
grid.addEventListener("pointerdown",function(e){const d=e.target.closest(".g");if(!d)return;dragging=+d.dataset.i;
  try{d.setPointerCapture(e.pointerId);}catch(x){} d.focus();move(e);e.preventDefault();});
grid.addEventListener("pointermove",function(e){if(dragging!==null)move(e);});
function endDrag(){if(dragging!==null){dragging=null;commit();}}
grid.addEventListener("pointerup",endDrag);grid.addEventListener("pointercancel",endDrag);
grid.addEventListener("dblclick",function(e){const d=e.target.closest(".g");if(!d)return;const i=+d.dataset.i,g=args.gauges[i];
  state[g.key]=g.value;update(i);commit();});
grid.addEventListener("keydown",function(e){const d=e.target.closest(".g");if(!d)return;
  const i=+d.dataset.i,g=args.gauges[i];let dt=0;
  if(e.key==="ArrowRight"||e.key==="ArrowUp")dt=0.02;else if(e.key==="ArrowLeft"||e.key==="ArrowDown")dt=-0.02;else return;
  const t=Math.min(1,Math.max(0,tOf(g,state[g.key])+dt*(e.shiftKey?5:1)));
  let v=vOf(g,t);if(v===state[g.key]){v=Math.min(g.max,Math.max(g.min,state[g.key]+(dt>0?g.step:-g.step)));}
  state[g.key]=v;update(i);e.preventDefault();clearTimeout(timer);timer=setTimeout(commit,350);});
window.addEventListener("message",function(e){
  if(!e.data||e.data.type!=="streamlit:render")return;
  args=e.data.args;
  if(args.token!==token){token=args.token;state={};args.gauges.forEach(function(g){state[g.key]=g.value;});lastSent=JSON.stringify(state);}
  build();
});
new ResizeObserver(fit).observe(grid);
send("streamlit:componentReady",{apiVersion:1});
</script></body></html>
"""


@st.cache_resource(show_spinner=False)
def get_gauge_component():
    for base in (BASE_DIR / ".cv_gauge", Path(os.environ.get("TMPDIR", "/tmp")) / "cv_gauge"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            f = base / "index.html"
            if not f.exists() or f.read_text(encoding="utf-8") != GAUGE_HTML:
                f.write_text(GAUGE_HTML, encoding="utf-8")
            return components.declare_component("cv_gauges", path=str(base))
        except Exception:
            continue
    raise RuntimeError("Impossible de créer le composant des compteurs.")


def gauge_panel(keys, cols, key, init, token):
    T = theme()
    specs = []
    for k in keys:
        s = dict(GAUGE_SPECS[k])
        s["key"] = k
        s["value"] = init[k]
        specs.append(s)
    tj = {k: T[k] for k in ["accent", "accent2", "track", "tick", "text", "muted", "cell", "border", "needle"]}
    comp = get_gauge_component()
    out = comp(gauges=specs, cols=cols, theme=tj, token=token, key=key, default=init)
    return out if isinstance(out, dict) and out else init


# ----------------------------------------------------------------------------
# PIPELINE ML (logique du notebook)
# ----------------------------------------------------------------------------
def build_features(raw, columns=None):
    """Prépare X comme dans le notebook (suppression des identifiants, get_dummies)."""
    raw = raw.copy()
    num = raw.select_dtypes(include="number").columns
    raw[num] = raw[num].fillna(0)
    X = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns])
    if "dst_port" in X.columns:
        X["dst_port"] = pd.to_numeric(X["dst_port"], errors="coerce").fillna(0).astype(int)
    cat = [c for c in ["protocol", "dst_port"] if c in X.columns]
    X = pd.get_dummies(X, columns=cat, dtype=int)
    X = X.fillna(0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def arrondir(scores, n):
    return np.clip(np.rint(scores), 0, n - 1).astype(int)


@st.cache_resource(show_spinner="Chargement du dataset et entraînement des modèles...")
def run_pipeline(csv_path, signature):
    t0 = time.time()
    df = pd.read_csv(csv_path, on_bad_lines="skip").dropna(subset=[TARGET])
    n_loaded = len(df)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    nan_series = df.isna().sum()
    nan_cols = nan_series[nan_series > 0].sort_values(ascending=False)
    n_nan = int(nan_series.sum())
    num_all = df.select_dtypes(include="number").columns
    df[num_all] = df[num_all].fillna(0)
    df = df.reset_index(drop=True)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[TARGET]), index=df.index)
    classes = list(le.classes_)
    nb = len(classes)
    X = build_features(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_val)
    Xte_s = scaler.transform(X_test)

    models, preds = {}, {}
    m = LinearRegression().fit(Xtr_s, y_train)
    models["Régression linéaire"] = m
    preds["Régression linéaire"] = arrondir(m.predict(Xva_s), nb)
    m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xtr_s, y_train)
    models["Régression logistique"] = m
    preds["Régression logistique"] = m.predict(Xva_s)
    m = SVC(kernel="rbf", C=1.0).fit(Xtr_s, y_train)
    models["SVM"] = m
    preds["SVM"] = m.predict(Xva_s)
    m = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    models["Random Forest"] = m
    preds["Random Forest"] = m.predict(X_val)
    m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    models["XGBoost"] = m
    preds["XGBoost"] = m.predict(X_val)

    rows = []
    for nom, yp in preds.items():
        rows.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, yp),
            "Précision macro": precision_score(y_val, yp, average="macro", zero_division=0),
            "Recall macro": recall_score(y_val, yp, average="macro", zero_division=0),
            "F1-score macro": f1_score(y_val, yp, average="macro", zero_division=0),
        })
    table = pd.DataFrame(rows).sort_values("F1-score macro", ascending=False).reset_index(drop=True)
    best = table.iloc[0]["Modèle"]

    def predict_with(name, Xraw, Xscaled):
        mod = models[name]
        if name in NO_SCALING:
            out = mod.predict(Xraw)
        else:
            out = mod.predict(Xscaled)
        if name == "Régression linéaire":
            out = arrondir(out, nb)
        return out

    pred_test = predict_with(best, X_test, Xte_s)
    cm = confusion_matrix(y_test, pred_test, labels=list(range(nb)))
    report = classification_report(y_test, pred_test, target_names=classes, output_dict=True, zero_division=0)
    spec = []
    for k in range(nb):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    active = best if best in MODELS_WITH_PROBA else "XGBoost"
    Xte_for = Xte_s if active not in NO_SCALING else X_test.values
    proba_test = models[active].predict_proba(Xte_for)
    try:
        auc = float(roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    metrics = dict(
        accuracy=accuracy_score(y_test, pred_test),
        precision=precision_score(y_test, pred_test, average="macro", zero_division=0),
        recall=recall_score(y_test, pred_test, average="macro", zero_division=0),
        f1=f1_score(y_test, pred_test, average="macro", zero_division=0),
        specificity=float(np.mean(spec)), auc=auc,
    )

    # Journal des prédictions (jeu de test)
    pt = models[active].predict(Xte_for)
    inc = df.loc[X_test.index, ["timestamp_utc", "src_ip", "dst_ip", "dst_port", "protocol", TARGET]].copy()
    inc["predite"] = [classes[i] for i in pt]
    inc["proba"] = proba_test.max(axis=1)
    inc["ts"] = pd.to_datetime(inc["timestamp_utc"], errors="coerce")
    inc = inc.sort_values("ts", ascending=False)

    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("sample_id", "is_attack", "src_port", "dst_port")]
    importances = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False)

    # Temps de prédiction moyen (ms / flux)
    t1 = time.time()
    _ = models[active].predict_proba(Xte_for[:500])
    ms_per = (time.time() - t1) / 500 * 1000

    try:
        os.makedirs(BASE_DIR / "models", exist_ok=True)
        joblib.dump({"model": models[best], "model_name": best, "scaler": scaler,
                     "needs_scaling": best not in NO_SCALING, "columns": list(X.columns), "classes": classes},
                    BASE_DIR / "models" / "best_model.pkl")
        table.round(4).to_csv(BASE_DIR / "models" / "results.csv", index=False)
    except Exception:
        pass

    return dict(
        df=df, classes=classes, columns=list(X.columns), scaler=scaler, models=models, table=table, best=best,
        active=active, metrics=metrics, cm=cm, report=report, inc=inc, num_cols=num_cols,
        importances=importances, n_loaded=n_loaded, n_dup=n_dup, n_nan=n_nan, nan_cols=nan_cols,
        sizes=(len(X_train), len(X_val), len(X_test)), n_features=X.shape[1],
        gmed=df[num_cols].median(), cmed=df.groupby(TARGET)[num_cols].median(),
        protocols=sorted(df["protocol"].astype(str).unique()),
        ports=sorted(int(p) for p in df["dst_port"].unique()),
        mode_proto=df.groupby(TARGET)["protocol"].agg(lambda s: s.mode().iloc[0]).to_dict(),
        mode_port=df.groupby(TARGET)["dst_port"].agg(lambda s: int(s.mode().iloc[0])).to_dict(),
        ms_per=ms_per, train_time=time.time() - t0,
        y_test_names=[classes[i] for i in y_test],
    )


def predict_proba_rows(P, model_name, raw_df):
    X = build_features(raw_df, P["columns"])
    mod = P["models"][model_name]
    Xin = P["scaler"].transform(X) if model_name not in NO_SCALING else X.values
    return mod.predict_proba(Xin)


def clip_round(v, g):
    v = min(g["max"], max(g["min"], float(v)))
    v = round(v / g["step"]) * g["step"]
    return round(v, 2)


def active_model(P):
    name = st.session_state.get("model_name", P["active"])
    return name if name in P["models"] and name in MODELS_WITH_PROBA else P["active"]


def run_prediction(P, scen, vals, proto, port):
    base = P["gmed"] if scen == GLOBAL_SCENARIO else P["cmed"].loc[scen]
    d = {c: float(base[c]) for c in P["num_cols"]}
    d.update({k: float(v) for k, v in vals.items()})
    d["protocol"] = proto
    d["dst_port"] = int(port)
    name = active_model(P)
    t0 = time.time()
    proba = predict_proba_rows(P, name, pd.DataFrame([d]))[0]
    ms = (time.time() - t0) * 1000
    order = np.argsort(proba)[::-1]
    return dict(proba=proba, top=[(P["classes"][i], float(proba[i])) for i in order[:3]],
                pred=P["classes"][order[0]], conf=float(proba[order[0]]), model=name, ms=ms)


def result_html(res, compact=False, top3=True):
    conf = res["conf"]
    cl = res["pred"]
    state = "Attaque confirmée" if conf >= 0.9 else "Attaque probable"
    rows = ""
    for name, p in res["top"]:
        rows += (f"<div class='r'><span><span class='dotc' style='background:{ccolor(name)}'></span>{disp(name)}</span>"
                 f"<div class='b'><i style='width:{max(p * 100, 0.4):.2f}%;background:{ccolor(name)}'></i></div>"
                 f"<span class='p'>{pct(p)}</span></div>")
    return (f"<div class='res'><div class='res-l'><div class='res-ring'>{ico('shieldx')}</div>"
            f"<div><div class='res-k'>Résultat de la prédiction</div><div class='res-v'>{'Attaque' if compact else state}</div></div></div>"
            f"<div class='res-r'><div class='res-k'>Confiance</div><div class='res-c'>{pct(conf)}</div>"
            f"<div class='res-bar'><i style='width:{conf * 100:.2f}%'></i></div></div></div>"
            f"<div class='res-meta'><span>Classe prédite</span><span class='chip' style='background:{ccolor(cl)}'>{disp(cl)}</span>"
            f"<span>Modèle : <b>{res['model']}</b></span><span>Temps : <b>{fr(res['ms'], 0)}\u00a0ms</b></span></div>"
            + (f"<div class='res-meta' style='margin-bottom:-4px'><span>Probabilités (Top 3)</span></div>"
               f"<div class='top3'>{rows}</div>" if top3 else ""))


# ----------------------------------------------------------------------------
# PANNEAU DE TEST (compteurs de vitesse)
# ----------------------------------------------------------------------------
def test_panel(P, prefix, keys, cols, compact):
    scen_opts = [GLOBAL_SCENARIO] + P["classes"]
    scen = st.selectbox("Scénario préréglé", scen_opts, format_func=lambda c: c if c == GLOBAL_SCENARIO else disp(c),
                        key=f"{prefix}_scen")
    c1, c2 = st.columns(2)
    d_proto = P["mode_proto"].get(scen, "TCP")
    d_port = P["mode_port"].get(scen, 443)
    proto = c1.selectbox("Protocole", P["protocols"], index=P["protocols"].index(d_proto) if d_proto in P["protocols"] else 0,
                         key=f"{prefix}_proto_{scen}")
    port = c2.selectbox("Port de destination", P["ports"], index=P["ports"].index(d_port) if d_port in P["ports"] else 0,
                        key=f"{prefix}_port_{scen}")
    base = P["gmed"] if scen == GLOBAL_SCENARIO else P["cmed"].loc[scen]
    init = {k: clip_round(base[k], GAUGE_SPECS[k]) for k in keys}
    vals = gauge_panel(keys, cols, f"g_{prefix}_{scen}", init, token=f"{prefix}|{scen}")
    if st.button("Lancer la prédiction", type="primary", width="stretch", key=f"{prefix}_btn"):
        st.session_state[f"res_{prefix}"] = run_prediction(P, scen, vals, proto, port)
    return st.session_state.get(f"res_{prefix}")


# ----------------------------------------------------------------------------
# AUTHENTIFICATION
# ----------------------------------------------------------------------------
def check_credentials(u, p):
    ok_u = hmac.compare_digest(u.encode("utf-8"), USERNAME.encode("utf-8"))
    ok_p = hmac.compare_digest(p.encode("utf-8"), PASSWORD.encode("utf-8"))
    return ok_u and ok_p


def login_page():
    inject_css()
    st.markdown("<style>section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl']{display:none !important}</style>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        html(f"<div class='login-hero'>{LOGO_SVG}<div class='n'>Cyber<b>Vion</b></div><div class='a'>ANALYTICS</div>"
             f"<div class='d'>Plateforme intelligente de détection et de prédiction des cyberattaques</div></div>")
        with st.container(key="card_login"):
            with st.form("login", border=False):
                u = st.text_input("Nom d'utilisateur", placeholder="Identifiant")
                p = st.text_input("Mot de passe", type="password", placeholder="Mot de passe")
                go_ = st.form_submit_button("Se connecter", type="primary", width="stretch")
            if go_:
                if check_credentials(u, p):
                    st.session_state["auth"] = True
                    st.session_state["user"] = u
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
def header():
    user = st.session_state.get("user", USERNAME)
    html(f"""<div class='cv-head'><div class='cv-head-l'><div class='cv-shield'>{ico('shieldcheck')}</div>
    <div><div class='cv-title'>Prédiction des attaques classiques en cyber</div>
    <div class='cv-sub'>Analyse du trafic<i>&bull;</i>Prédiction des attaques<i>&bull;</i>Évaluation du modèle<i>&bull;</i>Test de scénarios</div></div></div>
    <div class='cv-user'><div class='cv-av'>{user[:1].upper()}</div><div><div class='cv-un'>{user}</div><div class='cv-ur'>Analyste sécurité</div></div></div></div>""")


def page_title(t, s):
    html(f"<div class='pg-title'>{t}</div><div class='pg-sub'>{s}</div>")


def kpi(icon, label, value, sub):
    return (f"<div class='kpi'><div class='kpi-i'>{ico(icon)}</div><div><div class='kpi-l'>{label}</div>"
            f"<div class='kpi-v'>{value}</div><div class='kpi-s'>{sub}</div></div></div>")


def metric_cards(P):
    m = P["metrics"]
    html("<div class='kpis'>" + "".join([
        kpi("flux", "Flux analysés", fr(len(P["df"]), 0), f"{P['n_features']} variables"),
        kpi("layers", "Classes surveillées", str(len(P["classes"])), "attaques distinctes"),
        kpi("target", "Accuracy", pct(m["accuracy"]), "jeu de test"),
        kpi("shieldcheck", "Précision", pct(m["precision"]), "moyenne macro"),
        kpi("chart", "Recall", pct(m["recall"]), "moyenne macro"),
        kpi("gauge", "F1-score", pct(m["f1"]), f"modèle : {P['best']}"),
    ]) + "</div>")


def page_dashboard(P):
    metric_cards(P)
    T = theme()
    left, right = st.columns([2.05, 1], gap="medium")
    df, inc, m = P["df"], P["inc"], P["metrics"]

    with left:
        c1, c2 = st.columns([1.55, 1], gap="medium")
        with c1:
            with st.container(key="card_trend"):
                card_title("flux", "Évolution temporelle des attaques", "jeu de test")
                d = inc.dropna(subset=["ts"]).copy()
                d["mois"] = d["ts"].dt.month
                real = d.groupby("mois").size()
                ok = d[d[TARGET] == d["predite"]].groupby("mois").size().reindex(real.index, fill_value=0)
                xs = [MONTHS_FR[i - 1] for i in real.index]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=real.values, name="Attaques réelles", mode="lines+markers",
                                         line=dict(color=T["accent2"], width=2.5, shape="spline"),
                                         fill="tozeroy", fillcolor="rgba(34,211,238,0.10)", marker=dict(size=6)))
                fig.add_trace(go.Scatter(x=xs, y=ok.values, name="Détections correctes", mode="lines+markers",
                                         line=dict(color=T["accent"], width=2.5, shape="spline", dash="dot"),
                                         marker=dict(size=6)))
                style_fig(fig, 250)
                fig.update_layout(legend=dict(orientation="h", y=1.14, x=0))
                fig.update_yaxes(rangemode="tozero")
                show(fig, "dash_trend")
        with c2:
            with st.container(key="card_perf"):
                card_title("gauge", "Performances du modèle")
                fig = go.Figure(go.Pie(values=[m["accuracy"], max(1e-3, 1 - m["accuracy"])], hole=0.78, sort=False,
                                       marker=dict(colors=[T["accent"], T["track"]]), textinfo="none", hoverinfo="skip",
                                       direction="clockwise"))
                fig.add_annotation(text=f"<b>{pct(m['accuracy'])}</b><br><span style='font-size:11px'>Accuracy</span>",
                                   showarrow=False, font=dict(size=21, color=T["text"], family=FONT))
                style_fig(fig, 150, legend=False)
                show(fig, "dash_perf")
                html("".join(
                    f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:3px 2px'>"
                    f"<span style='color:{T['muted']}'>{n}</span><b>{v}</b></div>"
                    for n, v in [("Recall", pct(m["recall"])), ("Spécificité", pct(m["specificity"])),
                                 ("F1-score", pct(m["f1"])), ("AUC (ROC)", pct(m["auc"]))]))
        c3, c4 = st.columns([1.3, 1], gap="medium")
        counts = df[TARGET].value_counts()
        with c3:
            with st.container(key="card_freq"):
                card_title("chart", "Fréquence des attaques par classe")
                cs = counts.sort_values()
                fig = go.Figure(go.Bar(x=cs.values, y=[disp(c) for c in cs.index], orientation="h",
                                       marker=dict(color=[ccolor(c) for c in cs.index]),
                                       text=[fr(v, 0) for v in cs.values], textposition="outside",
                                       textfont=dict(color=T["text"], size=10), cliponaxis=False))
                style_fig(fig, 300, legend=False)
                fig.update_xaxes(range=[0, cs.max() * 1.18], showgrid=True)
                fig.update_yaxes(tickfont=dict(color=T["text"], size=11))
                show(fig, "dash_freq")
        with c4:
            with st.container(key="card_rep"):
                card_title("pie", "Répartition des attaques", "10 classes")
                cs = counts.sort_index()
                fig = go.Figure(go.Pie(labels=[disp(c) for c in cs.index], values=cs.values, hole=0.62,
                                       marker=dict(colors=[ccolor(c) for c in cs.index], line=dict(color=T["bg"], width=1.5)),
                                       textinfo="none", sort=False))
                fig.add_annotation(text=f"<b>{fr(cs.sum(), 0)}</b><br><span style='font-size:10px'>flux</span>", showarrow=False,
                                   font=dict(size=17, color=T["text"], family=FONT))
                style_fig(fig, 300)
                fig.update_layout(legend=dict(font=dict(size=10, color=T["text"]), x=1.0, y=0.5))
                show(fig, "dash_rep")
        with st.container(key="card_inc"):
            card_title("alert", "Incidents récents (prédictions)", "5 derniers flux du jeu de test")
            rows = []
            for i, (_, r) in enumerate(inc.head(5).iterrows(), start=1):
                rows.append([i, str(r["timestamp_utc"]).replace("T", " "), r["src_ip"], r["dst_ip"],
                             f"<span class='dotc' style='background:{ccolor(r['predite'])}'></span>{disp(r['predite'])}",
                             pct(r["proba"]), "<span class='badge'>Attaque</span>"])
            html(html_table(["#", "Horodatage", "Adresse source", "Adresse destination", "Classe prédite", "Probabilité", "Résultat"], rows))

    with right:
        with st.container(key="card_test"):
            card_title("flask", "Partie de test", "modifiez les compteurs")
            res = test_panel(P, "dash", GAUGES_COMPACT, 2, True)
            if res:
                html(result_html(res, compact=True))
            else:
                html("<div class='empty'>Réglez les compteurs puis lancez la prédiction.</div>")


def page_predict(P):
    page_title("Prédiction / Test de scénarios", "Tournez les compteurs (glisser, cliquer ou flèches du clavier), double-clic pour réinitialiser.")
    left, right = st.columns([2.3, 1], gap="medium")
    with left:
        with st.container(key="card_testfull"):
            card_title("flask", "Paramètres du trafic", "16 compteurs")
            res = test_panel(P, "full", GAUGES_FULL, 4, False)
    with right:
        with st.container(key="card_diag"):
            card_title("shieldcheck", "Diagnostic IA")
            if res:
                html(result_html(res, compact=False, top3=False))
                T = theme()
                order = np.argsort(res["proba"])[::-1]
                fig = go.Figure(go.Bar(x=[res["proba"][i] * 100 for i in order][::-1],
                                       y=[disp(P["classes"][i]) for i in order][::-1], orientation="h",
                                       marker=dict(color=[ccolor(P["classes"][i]) for i in order][::-1])))
                style_fig(fig, 300, legend=False)
                fig.update_xaxes(title=None, ticksuffix=" %", range=[0, 100])
                fig.update_yaxes(tickfont=dict(color=T["text"], size=10.5))
                show(fig, "pred_bar")
                lab, desc, act = CLASS_INFO[res["pred"]]
                html(f"<div class='note'><b>Interprétation.</b> {lab} : {desc}</div>"
                     f"<div class='note'><b>Action recommandée.</b> {act}</div>")
            else:
                html("<div class='empty'>Aucune prédiction pour le moment.<br>Réglez les compteurs puis cliquez sur « Lancer la prédiction ».</div>")


def page_analyses(P):
    page_title("Analyses", "Exploration statistique des variables réseau selon le type d'attaque.")
    df, T = P["df"], theme()
    t1, t2, t3, t4 = st.tabs(["Distribution des variables", "Profil par attaque", "Corrélations", "Importance des variables"])
    with t1:
        with st.container(key="card_an1"):
            c1, c2 = st.columns([3, 1])
            feat = c1.selectbox("Variable", P["num_cols"], index=P["num_cols"].index("packet_rate_pps"))
            logy = c2.toggle("Échelle logarithmique", value=True)
            s = df.sample(min(len(df), 12000), random_state=1)
            fig = go.Figure()
            for c in P["classes"]:
                fig.add_trace(go.Box(y=s.loc[s[TARGET] == c, feat], name=disp(c), marker_color=ccolor(c),
                                     boxpoints=False, line=dict(width=1.5)))
            style_fig(fig, 430, legend=False)
            if logy:
                fig.update_yaxes(type="log")
            fig.update_xaxes(tickfont=dict(color=T["text"], size=10))
            show(fig, "an_box")
    with t2:
        with st.container(key="card_an2"):
            card_title("layers", "Médiane normalisée des variables clés par attaque")
            keys = [k for k in GAUGES_FULL if k in P["cmed"].columns]
            med = P["cmed"][keys]
            z = (med - med.min()) / (med.max() - med.min()).replace(0, 1)
            fig = go.Figure(go.Heatmap(
                z=z.values, x=[GAUGE_SPECS[k]["label"] for k in keys], y=[disp(c) for c in med.index],
                text=[[fr(v, 1) for v in r] for r in med.values], texttemplate="%{text}", textfont=dict(size=10),
                colorscale=[[0, "rgba(59,130,246,0.05)"], [0.5, "#3b82f6"], [1, "#22d3ee"]] if T["name"] == "dark"
                else [[0, "#f1f5ff"], [0.5, "#818cf8"], [1, "#4338ca"]], showscale=False))
            style_fig(fig, 460, legend=False)
            fig.update_yaxes(autorange="reversed", tickfont=dict(color=T["text"], size=11))
            fig.update_xaxes(tickangle=-35, tickfont=dict(color=T["text"], size=10))
            show(fig, "an_prof")
    with t3:
        with st.container(key="card_an3"):
            card_title("grid", "Matrice de corrélation")
            corr = df[P["num_cols"]].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                                       colorscale="RdBu", reversescale=True, colorbar=dict(thickness=10, tickfont=dict(color=T["muted"]))))
            style_fig(fig, 620, legend=False)
            fig.update_yaxes(autorange="reversed", tickfont=dict(size=9, color=T["muted"]))
            fig.update_xaxes(tickangle=-60, tickfont=dict(size=9, color=T["muted"]))
            show(fig, "an_corr")
    with t4:
        with st.container(key="card_an4"):
            card_title("chart", "Variables les plus déterminantes", "Random Forest")
            imp = P["importances"].head(20)[::-1]
            fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h",
                                   marker=dict(color=imp.values, colorscale=[[0, T["accent"]], [1, T["accent2"]]])))
            style_fig(fig, 520, legend=False)
            fig.update_xaxes(ticksuffix=" %")
            fig.update_yaxes(tickfont=dict(color=T["text"], size=10.5))
            show(fig, "an_imp")


def page_incidents(P):
    page_title("Incidents", "Journal des prédictions et analyse d'un fichier CSV.")
    inc = P["inc"]
    t1, t2 = st.tabs(["Journal des prédictions", "Analyse d'un fichier CSV"])
    with t1:
        with st.container(key="card_i1"):
            c1, c2, c3 = st.columns([2, 1, 1])
            sel = c1.multiselect("Classe prédite", P["classes"], format_func=disp, placeholder="Toutes les classes")
            thr = c2.selectbox("Probabilité minimale", [0, 50, 80, 90, 95, 99], format_func=lambda v: f"{v} %")
            nmax = c3.selectbox("Lignes affichées", [25, 50, 100, 200], index=1)
            d = inc
            if sel:
                d = d[d["predite"].isin(sel)]
            d = d[d["proba"] * 100 >= thr]
            st.caption(f"{fr(len(d), 0)} incidents correspondent aux filtres.")
            rows = []
            for i, (_, r) in enumerate(d.head(nmax).iterrows(), start=1):
                good = r["predite"] == r[TARGET]
                rows.append([i, str(r["timestamp_utc"]).replace("T", " "), r["src_ip"], r["dst_ip"], r["protocol"], int(r["dst_port"]),
                             disp(r[TARGET]),
                             f"<span class='dotc' style='background:{ccolor(r['predite'])}'></span>{disp(r['predite'])}",
                             pct(r["proba"]),
                             "<span class='badge'>Attaque</span>" if good else "<span class='badge ok'>Écart</span>"])
            html(html_table(["#", "Horodatage", "Source", "Destination", "Proto.", "Port", "Classe réelle", "Classe prédite", "Probabilité", "Résultat"], rows))
            exp = d[["timestamp_utc", "src_ip", "dst_ip", "protocol", "dst_port", TARGET, "predite", "proba"]]
            st.download_button("Télécharger le journal (CSV)", exp.to_csv(index=False).encode("utf-8"),
                               "journal_predictions.csv", "text/csv")
    with t2:
        with st.container(key="card_i2"):
            up = st.file_uploader("Déposez un fichier CSV de flux réseau (mêmes colonnes que le dataset)", type=["csv"])
            if up is not None:
                try:
                    raw = pd.read_csv(up, on_bad_lines="skip")
                    name = active_model(P)
                    proba = predict_proba_rows(P, name, raw)
                    pred = [P["classes"][i] for i in proba.argmax(axis=1)]
                    out = raw.copy()
                    out["classe_predite"] = pred
                    out["probabilite"] = proba.max(axis=1)
                    st.success(f"{fr(len(out), 0)} flux analysés avec le modèle {name}.")
                    vc = pd.Series(pred).value_counts()
                    T = theme()
                    fig = go.Figure(go.Bar(x=[disp(c) for c in vc.index], y=vc.values, marker=dict(color=[ccolor(c) for c in vc.index])))
                    style_fig(fig, 280, legend=False)
                    fig.update_xaxes(tickfont=dict(color=T["text"]))
                    show(fig, "up_bar")
                    if TARGET in raw.columns:
                        st.caption(f"Accuracy sur ce fichier : {pct(float((raw[TARGET] == out['classe_predite']).mean()))}")
                    st.download_button("Télécharger les prédictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                                       "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Fichier non exploitable : {e}")


def page_data(P):
    page_title("Données & Modèle", "Jeu de données, prétraitement et comparaison des algorithmes.")
    df, T = P["df"], theme()
    ntr, nva, nte = P["sizes"]
    html("<div class='kpis'>" + "".join([
        kpi("db", "Lignes chargées", fr(P["n_loaded"], 0), "fichier dataset.csv"),
        kpi("layers", "Colonnes", str(df.shape[1]), f"{P['n_features']} variables après encodage"),
        kpi("list", "Doublons supprimés", fr(P["n_dup"], 0), f"{fr(len(df), 0)} lignes conservées"),
        kpi("alert", "Valeurs manquantes", fr(P["n_nan"], 0), "remplacées par 0"),
        kpi("chart", "Train / Val / Test", f"{fr(ntr / 1000, 1)} / {fr(nva / 1000, 1)} / {fr(nte / 1000, 1)} k", "70 % / 15 % / 15 %"),
        kpi("clock", "Entraînement", f"{fr(P['train_time'], 1)}\u00a0s", "5 modèles"),
    ]) + "</div>")
    with st.container(key="card_d1"):
        card_title("db", "Aperçu du dataset")
        show_cols = ["timestamp_utc", "src_ip", "dst_ip", "protocol", "dst_port", "packet_rate_pps",
                     "flow_duration_ms", "payload_entropy", "unique_dst_ports_5m", "failed_logins_5m", TARGET]
        rows = [[(f"<span class='dotc' style='background:{ccolor(v)}'></span>{disp(v)}" if c == TARGET else
                  (fr(v, 2) if isinstance(v, float) else v)) for c, v in zip(show_cols, r)]
                for r in df[show_cols].head(12).values.tolist()]
        html(html_table(show_cols, rows))
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with st.container(key="card_d2"):
            card_title("chart", "Comparaison des modèles", "jeu de validation")
            tb = P["table"]
            fig = go.Figure(go.Bar(x=tb["Modèle"], y=tb["F1-score macro"],
                                   marker=dict(color=[T["accent2"] if m == P["best"] else T["accent"] for m in tb["Modèle"]]),
                                   text=[pct(v, 2) for v in tb["F1-score macro"]], textposition="outside",
                                   textfont=dict(color=T["text"], size=10), cliponaxis=False))
            style_fig(fig, 300, legend=False)
            fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
            fig.update_xaxes(tickfont=dict(color=T["text"], size=10))
            show(fig, "data_cmp")
    with c2:
        with st.container(key="card_d3"):
            card_title("list", "Scores sur validation")
            tb = P["table"]
            rows = [[("<b>" + r["Modèle"] + "</b>" if r["Modèle"] == P["best"] else r["Modèle"]),
                     pct(r["Accuracy"], 3), pct(r["Précision macro"], 3), pct(r["Recall macro"], 3), pct(r["F1-score macro"], 3)]
                    for _, r in tb.iterrows()]
            html(html_table(["Modèle", "Accuracy", "Précision", "Recall", "F1-score"], rows))
            html(f"<div class='note'>Meilleur modèle sélectionné sur le F1-score macro : <b>{P['best']}</b>. "
                 "Le scaler est ajusté uniquement sur le jeu d'entraînement afin d'éviter toute fuite de données.</div>")
    with st.container(key="card_d4"):
        card_title("list", "Statistiques descriptives", "variables numériques clés")
        desc = df[GAUGES_FULL].describe().T
        rows = [[GAUGE_SPECS[k]["label"], fr(r["mean"], 2), fr(r["std"], 2), fr(r["min"], 2), fr(r["50%"], 2), fr(r["max"], 2)]
                for k, r in desc.iterrows()]
        html(html_table(["Variable", "Moyenne", "Écart-type", "Min", "Médiane", "Max"], rows))


def page_eval(P):
    page_title("Évaluation", f"Résultats du modèle {P['best']} sur le jeu de test (données non vues).")
    T, m = theme(), P["metrics"]
    html("<div class='kpis' style='grid-template-columns:repeat(4,minmax(0,1fr))'>" + "".join([
        kpi("target", "Accuracy", pct(m["accuracy"]), "taux global de bonnes prédictions"),
        kpi("shieldcheck", "Precision", pct(m["precision"]), "moyenne macro"),
        kpi("chart", "Recall", pct(m["recall"]), "moyenne macro"),
        kpi("gauge", "F1-score", pct(m["f1"]), "moyenne macro"),
    ]) + "</div>")
    c1, c2 = st.columns([1.25, 1], gap="medium")
    names = [disp(c) for c in P["classes"]]
    with c1:
        with st.container(key="card_e1"):
            card_title("grid", "Matrice de confusion", "lignes : vraie classe, colonnes : classe prédite")
            cm = P["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, text=cm, texttemplate="%{text}", textfont=dict(size=11),
                                       colorscale=[[0, "rgba(59,130,246,0.04)"], [1, T["accent"]]] if T["name"] == "dark"
                                       else [[0, "#f3f6ff"], [1, "#3b4cca"]], showscale=False))
            style_fig(fig, 520, legend=False)
            fig.update_yaxes(autorange="reversed", tickfont=dict(color=T["text"], size=10.5))
            fig.update_xaxes(tickangle=-40, tickfont=dict(color=T["text"], size=10.5))
            show(fig, "ev_cm")
    with c2:
        with st.container(key="card_e2"):
            card_title("list", "Rapport par classe")
            rows = []
            for c in P["classes"]:
                r = P["report"][c]
                rows.append([f"<span class='dotc' style='background:{ccolor(c)}'></span>{disp(c)}", pct(r["precision"]),
                             pct(r["recall"]), pct(r["f1-score"]), int(r["support"])])
            html(html_table(["Classe", "Précision", "Recall", "F1-score", "Support"], rows))
            html(f"<div class='note'>AUC (ROC) macro : <b>{pct(m['auc'])}</b> &nbsp;|&nbsp; Spécificité macro : <b>{pct(m['specificity'])}</b>. "
                 "Ce dataset est synthétique : de tels scores ne garantissent pas la même efficacité sur un réseau réel.</div>")


def page_settings(P):
    page_title("Paramètres", "Configuration de l'application et du modèle de prédiction.")
    with st.container(key="card_s1"):
        card_title("gauge", "Modèle de prédiction")
        cur = active_model(P)
        st.selectbox("Modèle utilisé pour les prédictions", MODELS_WITH_PROBA, index=MODELS_WITH_PROBA.index(cur), key="model_name")
        html(f"<div class='note'>Meilleur modèle sur validation : <b>{P['best']}</b>. "
             "Seuls les modèles fournissant des probabilités sont proposés pour le diagnostic.</div>")
    with st.container(key="card_s2"):
        card_title("db", "Informations")
        html(html_table(["Élément", "Valeur"], [
            ["Application", APP_NAME], ["Fichier de données", CSV_PATH.name],
            ["Flux utilisés", fr(len(P["df"]), 0)], ["Classes", ", ".join(disp(c) for c in P["classes"])],
            ["Temps de prédiction moyen", f"{fr(P['ms_per'], 2)}\u00a0ms / flux"],
        ]))
        html("<div class='note'>Le mode d'affichage (clair ou sombre) se règle depuis la barre latérale.</div>")


def sidebar():
    with st.sidebar:
        html(f"<div class='brand'>{LOGO_SVG}<div><div class='brand-t'>Cyber<b>Vion</b></div><div class='brand-s'>ANALYTICS</div></div></div>")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="page")
        html(f"<div class='side-card'>{ico('shield')}<div class='t'>IA de cybersécurité</div>"
             f"<div class='s'>Prédiction &bull; Détection<br>Classification &bull; Évaluation</div></div>")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.toggle("Mode sombre", value=True, key="dark_mode")
        if st.button("Se déconnecter", width="stretch", key="logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    return page


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    if not st.session_state.get("auth"):
        login_page()
        return
    page = sidebar()
    inject_css()
    if not CSV_PATH.exists():
        st.error(f"Fichier introuvable : {CSV_PATH}. Placez dataset.csv à côté de app.py.")
        st.stop()
    sig = f"{CSV_PATH.stat().st_size}-{int(CSV_PATH.stat().st_mtime)}"
    P = run_pipeline(str(CSV_PATH), sig)
    header()
    {"Tableau de bord": page_dashboard, "Prédiction / Test": page_predict, "Analyses": page_analyses,
     "Incidents": page_incidents, "Données & Modèle": page_data, "Évaluation": page_eval,
     "Paramètres": page_settings}[page](P)


main()
