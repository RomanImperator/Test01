"""
Near_Miss-SL_v06_03.py
======================
Pagina principale della WebApp "Near Miss Scuola – EduSafeBot".

Versione v06_03.5 – Fix CSV Generico + Parità Funzionale v06_02
Changelog rispetto a v06_03.4:
  - 🆕 MIGLIORIA NUOVA: Selectbox separatore CSV (;  ,  Tab  Auto) in sidebar
  - ⬅️ EREDITATA da v06_02: Normalizzazione colonna data in datetime (Blocco 13 originale)
  - ⬅️ EREDITATA da v06_02: Variabili ambiente HuggingFace/Tokenizer (Blocco 3 originale)
  - ✅ GIÀ PRESENTE: Doppio asse Y per CSV generico (UI in Utils righe 655-711)
  - ✅ GIÀ PRESENTE: Pannello RAG integrato in gestisci_chatbot()

Commenti in italiano.
"""

import os
import csv
import io
import streamlit as st
import pandas as pd

# ============================================================
# BLOCCO 1 – CONFIGURAZIONE PAGINA (DEVE ESSERE LA PRIMA CHIAMATA STREAMLIT)
# ============================================================
st.set_page_config(
    page_title="Near Miss Scuola – EduSafeBot",
    layout="wide",
    page_icon="🏫"
)

# ============================================================
# BLOCCO 2 – OTTIMIZZAZIONI AMBIENTE
# (⬅️ EREDITATA da v06_02 — Blocco 3 originale)
# Disattiva il parallelismo dei tokenizer e la telemetria HuggingFace
# per evitare warning e overhead inutili in ambiente Colab.
# ============================================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ============================================================
# BLOCCO 3 – IMPORT DAL PROGETTO
# ============================================================
# Importo le funzioni REALI da Utils_NearMiss_v06_03.py
from Utils_Dashboard import (
    _init_state,                # Inizializzazione stato sessione
    carica_dati,                # Caricamento CSV locale / Google Sheet
    google_sheet_available,     # Verifica rapida disponibilità Google Sheet
    read_google_with_timeout,   # Lettura Google Sheet con timeout
    visualizza_dashboard,       # Dashboard completa (tabella + grafici + doppio asse Y)
    seleziona_llm_sidebar,      # Selezione modello LLM in sidebar
    gestisci_chatbot,           # Chatbot RAG completo (pannello + form + chat)
    sync_csv_dashboard,         # Sincronizzazione CSV → RAG quando cambia fonte
)

# Importo la configurazione
from config import LLM_MODELS


# ============================================================
# BLOCCO 4 – INIZIALIZZAZIONE STATO DI SESSIONE
# ============================================================
# _init_state() inizializza chat_log, provider LLM, tipo_sorgente, ecc.
_init_state()


# ============================================================
# BLOCCO 5 – INTESTAZIONE DELLA WEBAPP
# ============================================================
st.title("🛡️ Near Miss Scuola – EduSafeBot")
st.markdown("""
**Come usare la pagina:**
1. Verifica che il CSV sia sincronizzato e l'indice RAG aggiornato (sezioni 📊 Dashboard e 📚 RAG).
2. Esplora i grafici per capire **dove**, **quando** e **come** avvengono i near miss.
3. Usa il **chatbot** per fare domande del tipo:
   - *"Quali sono le cause più frequenti di near miss?"*
   - *"Che tipo di interventi si potrebbero proporre in classe?"*
4. Durante le lezioni/laboratori, utilizza questa pagina come base per la discussione.

> ℹ️ Questa è una versione sperimentale/prototipale, pensata per progetti pilota e attività
> di ricerca/didattica. I risultati vanno sempre interpretati insieme ai docenti e ai referenti
> della sicurezza.
""")


# ============================================================
# BLOCCO 6 – MINI FEEDBACK DI BOOT DELL'INTERFACCIA
# (cosmetico, non influisce sulla logica)
# ============================================================
ph_boot = st.empty()
with ph_boot.container():
    st.info("⏳ Avvio interfaccia… caricamento componenti iniziali…")
ph_boot.empty()


# ============================================================
# BLOCCO 7 – SELEZIONE SORGENTE DATI (RADIO BUTTON ESCLUSIVO)
# (✅ MIGLIORIA v06_03 — sostituisce checkbox + expander v06_02)
# ============================================================
# La scelta della sorgente dati è ESCLUSIVA:
# selezionare un'opzione disabilita automaticamente le altre.

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Sorgente Dati")

# Opzioni disponibili per il selettore
OPZIONI_SORGENTE = [
    "💾 CSV Locale (Default)",
    "🔌 Google Sheet (Live)",
    "📂 Upload CSV Generico",
]

# Funzione di callback per sincronizzare la scelta immediatamente
def sync_sorgente():
    st.session_state["tipo_sorgente"] = st.session_state["radio_sorgente_key"]

# Radio button nella sidebar per scelta esclusiva
# ✅ MIGLIORIA v06_03.8: uso di on_change e key per eliminare il "rimbalzo" (doppio clic)
scelta_sorgente = st.sidebar.radio(
    "Seleziona l'origine dei dati:",
    options=OPZIONI_SORGENTE,
    index=OPZIONI_SORGENTE.index(st.session_state["tipo_sorgente"]),
    key="radio_sorgente_key",
    on_change=sync_sorgente
)


# ============================================================
# BLOCCO 8 – LOGICA DI CARICAMENTO DATI (ESCLUSIVA)
# ============================================================
# In base alla scelta del radio button, carico i dati dalla fonte selezionata.
# Se la fonte scelta non è disponibile, uso un fallback sul CSV locale.

# Variabili per il caricamento dati
df = None
fonte_dati = ""


# --- CASO 1: GOOGLE SHEET ---
if scelta_sorgente == "🔌 Google Sheet (Live)":
    # Verifico se il Google Sheet è raggiungibile
    disponibile, err_msg = google_sheet_available(timeout_sec=5)
    if disponibile:
        try:
            with st.spinner("⏳ Connessione a Google Sheet in corso…"):
                df, fonte_dati = read_google_with_timeout(10)
            if df is not None:
                st.success(f"✅ Dati sincronizzati da Google Sheet ({len(df)} record).")
            else:
                df, fonte_dati = carica_dati("csv")
        except Exception as e:
            st.error(f"❌ Errore durante il caricamento: {e}")
            df, fonte_dati = carica_dati("csv")
    else:
        st.error(f"❌ [v06_03.8.2] Google Sheet non raggiungibile. Motivo: {err_msg}")
        st.info("💡 Suggerimento: Verifica che il foglio sia impostato come **'Chiunque abbia il link può visualizzare'** su Google Sheets.")
        df, fonte_dati = carica_dati("csv")


# --- CASO 2: UPLOAD CSV GENERICO ---
elif scelta_sorgente == "📂 Upload CSV Generico":

    # ---------------------------------------------------------------
    # 🆕 MIGLIORIA NUOVA: Scelta del separatore CSV da parte dell'utente.
    # In Italia il separatore piu' comune e' il punto e virgola ";",
    # mentre lo standard internazionale e' la virgola ",".
    # L'opzione "Rileva automaticamente" usa csv.Sniffer per indovinare.
    # Questa funzionalita' NON era presente nella v06_02.
    # ---------------------------------------------------------------

    # Separatore visivo nella sidebar prima dei controlli CSV
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Impostazioni CSV")

    # Lista delle opzioni separatore (label ASCII-safe, senza caratteri speciali)
    LISTA_SEP_LABELS = [
        "; (Punto e virgola - Italia)",
        ", (Virgola - internazionale)",
        "Tab (Tabulazione)",
        "Rileva automaticamente",
    ]
    LISTA_SEP_VALORI = [";", ",", "\t", "auto"]

    scelta_sep_idx = st.sidebar.selectbox(
        "Separatore del CSV:",
        options=LISTA_SEP_LABELS,
        index=0,
        key="selectbox_separatore_csv",
    )
    # Ricavo il valore del separatore dalla posizione nella lista
    separatore_scelto = LISTA_SEP_VALORI[LISTA_SEP_LABELS.index(scelta_sep_idx)]

    # Il widget di upload appare subito sotto il selectbox
    uploaded_file = st.sidebar.file_uploader(
        "Scegli un file CSV dal tuo PC",
        type="csv",
        key="uploader_csv_generico",
    )

    if uploaded_file is not None:
        # Lettura del file caricato dall'utente
        try:
            uploaded_file.seek(0)
            contenuto_raw = uploaded_file.read()

            # Decodifica: provo UTF-8, poi latin-1 come fallback
            try:
                testo = contenuto_raw.decode("utf-8")
            except UnicodeDecodeError:
                testo = contenuto_raw.decode("latin-1")

            # Rilevo automaticamente il separatore se richiesto
            if separatore_scelto == "auto":
                try:
                    # csv.Sniffer analizza le prime righe per indovinare il delimitatore
                    campione = testo[:8192]
                    dialetto = csv.Sniffer().sniff(campione, delimiters=";,\t|")
                    sep_effettivo = dialetto.delimiter
                    st.info(f"Separatore rilevato automaticamente: `{repr(sep_effettivo)}`")
                except csv.Error:
                    sep_effettivo = ";"
                    st.warning("Rilevamento automatico fallito. Uso `;` come default.")
            else:
                sep_effettivo = separatore_scelto

            # Parsing del CSV con il separatore scelto/rilevato
            df = pd.read_csv(
                io.StringIO(testo),
                sep=sep_effettivo,
                engine="python",
                on_bad_lines="warn",
            )

            # Verifica: se il DataFrame ha una sola colonna, il separatore e' probabilmente errato
            if len(df.columns) <= 1 and len(testo.strip()) > 0:
                st.warning(
                    f"Il CSV ha solo **{len(df.columns)} colonna**. "
                    f"Il separatore `{repr(sep_effettivo)}` potrebbe non essere corretto. "
                    "Prova a cambiare il separatore nella selectbox in sidebar."
                )

            fonte_dati = f"CSV generico: {uploaded_file.name}"
            st.success(
                f"CSV generico caricato: **{uploaded_file.name}** - "
                f"**{len(df)}** righe x **{len(df.columns)}** colonne "
                f"(separatore: `{repr(sep_effettivo)}`)"
            )

        except Exception as e:
            st.error(f"Errore durante la lettura del file: {e}")
            df, fonte_dati = carica_dati("csv")
    else:
        # Nessun file ancora caricato: mostro avviso e uso il CSV locale come segnaposto
        st.warning("In attesa che venga caricato un file CSV...")
        df, fonte_dati = carica_dati("csv")
        st.info("Visualizzo i dati locali predefiniti finche' non carichi un file.")


# --- CASO 3: CSV LOCALE (DEFAULT) ---
else:
    df, fonte_dati = carica_dati("csv")
    st.success(f"✅ Utilizzo dati locali predefiniti ({len(df)} record).")


# ============================================================
# BLOCCO 9 – NORMALIZZAZIONE COLONNA DATA
# (⬅️ EREDITATA da v06_02 — Blocco 13 originale)
# Converte la colonna data in formato datetime per consentire
# filtri temporali e ordinamenti corretti.
# ============================================================
if df is not None and not df.empty:
    col_data = "📅 Data dell'evento Near Miss"
    if col_data in df.columns:
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce")


# ============================================================
# BLOCCO 10 – SINCRONIZZAZIONE RAG (quando cambia la fonte dati)
# (✅ MIGLIORIA v06_03 — automatizza la sync al cambio fonte)
# ============================================================
# Se la fonte dati è cambiata rispetto all'ultima volta, aggiorno
# il CSV nella cartella RAG e ricostruisco l'indice se necessario.
if df is not None:
    sync_csv_dashboard(df, fonte_dati)


# ============================================================
# BLOCCO 11 – TRACCIATO RECORD (CSV PREDEFINITO)
# (⬅️ EREDITATA da v06_02 — Blocco 11 originale)
# Mostra il tracciato record nella sidebar solo per il CSV locale.
# ============================================================
if scelta_sorgente == "💾 CSV Locale (Default)":
    from config import TRACCIATO_REC_DEFAULT_CSV
    from config import NOME_FILE_CSV_TRACCIATO_REC_DEFAULT

    trac_path = TRACCIATO_REC_DEFAULT_CSV
    if os.path.exists(trac_path):
        df_tracciato = pd.read_csv(trac_path, nrows=0)
        st.sidebar.markdown("### 📑 Tracciato record (CSV predefinito)")
        st.sidebar.dataframe(
            pd.DataFrame({"Colonne": df_tracciato.columns}),
            use_container_width=True,
        )
        st.sidebar.download_button(
            "⬇️ Scarica tracciato record",
            data=(",".join(df_tracciato.columns)).encode("utf-8"),
            file_name=NOME_FILE_CSV_TRACCIATO_REC_DEFAULT,
            mime="text/csv",
            key="btn_tracciato_record",
        )
    else:
        st.sidebar.warning("⚠️ Tracciato record non trovato nella cartella xls.")


# ============================================================
# BLOCCO 12 – SELEZIONE LLM IN SIDEBAR
# (⬅️ EREDITATA da v06_02 — Blocco 5 originale)
# seleziona_llm_sidebar restituisce (provider, embedding_model)
# ============================================================
provider, embedding_model = seleziona_llm_sidebar(LLM_MODELS)


# ============================================================
# BLOCCO 13 – DASHBOARD + CHATBOT (in Tab separate)
# ============================================================
# Dashboard:
#   - visualizza_dashboard(df) include: tabella filtrata + grafici
#   - Il DOPPIO ASSE Y per CSV generico è gestito internamente
#     nella sezione "CASO 3.2 — CSV GENERICO" di Utils (righe 655-711).
#     Appare come checkbox "Attiva asse Y secondario (Y2)".
#     (✅ GIÀ PRESENTE — ereditato da v06_02)
#
# Chatbot:
#   - gestisci_chatbot(provider, embedding_model) include:
#     pannello RAG + form domanda + storico chat
#     (✅ Integra il Blocco 12 del vecchio v06_02)

if df is not None:
    # Due tab principali: Dashboard e Chatbot
    tab_dash, tab_chat = st.tabs(["📊 Dashboard Statistica", "💬 Assistente AI (RAG)"])

    with tab_dash:
        # La dashboard include già la tabella, i filtri e tutti i grafici
        # con supporto al doppio asse Y per i CSV generici
        visualizza_dashboard(df)

    with tab_chat:
        # Il chatbot include già il pannello RAG con gestione indice FAISS
        gestisci_chatbot(provider, embedding_model)

else:
    st.error("❌ Errore critico: impossibile caricare i dati da nessuna fonte.")
