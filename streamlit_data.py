
import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Notes Réglementaires",
    page_icon="📝",
    layout="wide"
)

# Fichier de sauvegarde
DATA_FILE = "regulatory_notes.json"
HISTORY_FILE = "notes_history.json"

# Fonction pour charger les données
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "taiwan": "",
        "hong_kong": "",
        "australia": "",
        "last_updated": None
    }

# Fonction pour sauvegarder les données
def save_data(data):
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sauvegarder les données principales
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Ajouter à l'historique
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)

    history.append({
        "timestamp": data["last_updated"],
        "data": data.copy()
    })

    # Garder seulement les 50 dernières versions
    history = history[-50:]

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# En-tête de l'application
st.title("📝 Notes Réglementaires - Marchés Asiatiques")
st.markdown("---")

# Sidebar pour les options
with st.sidebar:
    st.header("⚙️ Options")

    # Statistiques
    data = load_data()
    if data["last_updated"]:
        st.info(f"📅 Dernière mise à jour: {data['last_updated']}")

    # Compteur de caractères
    st.subheader("📊 Statistiques")
    taiwan_chars = len(data.get("taiwan", ""))
    hk_chars = len(data.get("hong_kong", ""))
    aus_chars = len(data.get("australia", ""))

    st.metric("🇹🇼 Taiwan", f"{taiwan_chars} caractères")
    st.metric("🇭🇰 Hong Kong", f"{hk_chars} caractères")
    st.metric("🇦🇺 Australia", f"{aus_chars} caractères")

    st.markdown("---")

    # Export
    st.subheader("💾 Export")

    if st.button("📥 Télécharger JSON"):
        st.download_button(
            label="💾 Sauvegarder JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    if st.button("📊 Télécharger CSV"):
        df = pd.DataFrame([
            {"Marché": "Taiwan 🇹🇼", "Notes": data.get("taiwan", "")},
            {"Marché": "Hong Kong 🇭🇰", "Notes": data.get("hong_kong", "")},
            {"Marché": "Australia 🇦🇺", "Notes": data.get("australia", "")}
        ])
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 Sauvegarder CSV",
            data=csv,
            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Zone principale
# Charger les données existantes
data = load_data()

# Créer les colonnes pour une meilleure organisation
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🇹🇼 Taiwan")
    taiwan_note = st.text_area(
        "Notes pour Taiwan",
        value=data.get("taiwan", ""),
        height=300,
        key="taiwan_text",
        placeholder="Entrez vos notes sur les régulations de Taiwan..."
    )

with col2:
    st.subheader("🇭🇰 Hong Kong")
    hk_note = st.text_area(
        "Notes pour Hong Kong",
        value=data.get("hong_kong", ""),
        height=300,
        key="hk_text",
        placeholder="Entrez vos notes sur les régulations de Hong Kong..."
    )

with col3:
    st.subheader("🇦🇺 Australia")
    aus_note = st.text_area(
        "Notes pour Australia",
        value=data.get("australia", ""),
        height=300,
        key="aus_text",
        placeholder="Entrez vos notes sur les régulations de Australia..."
    )

# Boutons d'action
st.markdown("---")
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 2, 2, 4])

with col_btn1:
    if st.button("💾 Sauvegarder", type="primary", use_container_width=True):
        new_data = {
            "taiwan": taiwan_note,
            "hong_kong": hk_note,
            "australia": aus_note
        }
        save_data(new_data)
        st.success("✅ Données sauvegardées avec succès!")
        st.rerun()

with col_btn2:
    if st.button("🔄 Actualiser", use_container_width=True):
        st.rerun()

with col_btn3:
    if st.button("🗑️ Effacer tout", use_container_width=True):
        if st.session_state.get('confirm_delete', False):
            new_data = {
                "taiwan": "",
                "hong_kong": "",
                "australia": ""
            }
            save_data(new_data)
            st.session_state.confirm_delete = False
            st.success("🗑️ Toutes les notes ont été effacées")
            st.rerun()
        else:
            st.session_state.confirm_delete = True
            st.warning("⚠️ Cliquez à nouveau pour confirmer la suppression")

# Section d'historique (optionnelle)
with st.expander("📜 Historique des modifications"):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)

        if history:
            st.write(f"**{len(history)} versions sauvegardées**")

            for i, version in enumerate(reversed(history[-10:])):
                with st.container():
                    st.text(f"Version {len(history) - i}: {version['timestamp']}")

                    if st.button(f"Restaurer cette version", key=f"restore_{i}"):
                        save_data(version['data'])
                        st.success("✅ Version restaurée!")
                        st.rerun()

                    st.markdown("---")
        else:
            st.info("Aucun historique disponible")
    else:
        st.info("Aucun historique disponible")

# Footer
st.markdown("---")
st.markdown(
    """<div style='text-align: center'>
    <p>📝 Application de Notes Réglementaires | 
    Données sauvegardées localement | 
    © 2026</p>
    </div>""", 
    unsafe_allow_html=True
)
