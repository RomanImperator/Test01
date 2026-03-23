# =========================================================================
#  Configurazioni globali per Near Miss Scuola v06_03
#  Tutte le variabili centrali stanno qui, così:
#   - è più facile fare manutenzione
#   - è semplice cambiare modelli LLM / embedding
#   - si può spostare il progetto senza toccare la logica nei .py principali
#
#  Modifiche rispetto alla v06_02 (Fase 2 – Stabilizzazione):
#   - Fix #5: il percorso base del progetto (BASE_PATH) non è più fisso
#             ("scritto a mano nel codice"). Ora si può impostare da fuori
#             tramite una variabile d'ambiente NEARMISS_BASE_PATH.
#             Se non viene impostata, usa il percorso Colab di default.
#   - Fix #11: aggiunto un controllo all'avvio che verifica se la cartella
#              del progetto esiste davvero. Se non esiste (ad esempio perché
#              Google Drive non è collegato), lo segnala subito con un
#              messaggio chiaro, invece di dare errori incomprensibili più
#              avanti nel programma.
# =========================================================================

import os
from collections import OrderedDict
from pathlib import Path  # (non usato ovunque, ma utile se serve gestire path in modo portabile)

# =======================================================================
# ======================  SEZIONE: PERCORSI BASE  =======================
# =======================================================================
# --- Descrizione ---
# Qui definisci:
#  - la cartella principale del progetto su Google Drive
#  - i nomi dei file CSV “di default” (risposte dei moduli / tracciato record)
#  - i path completi usati dal codice per caricare i dati
# =======================================================================

# FIX #5 (v06_03): nella v06_02 questo percorso era scritto direttamente
# nel codice ("hardcoded"), il che significava che se Google Drive non era
# montato esattamente a quel percorso, oppure se si voleva usare la WebApp
# su un computer locale, bisognava modificare il sorgente ogni volta.
# Ora il programma cerca prima una variabile d'ambiente chiamata
# NEARMISS_BASE_PATH: se la trova, usa quel percorso. Se non la trova,
# usa il percorso Colab di default (che è comunque quello usuale).
BASE_PATH = os.getenv("NEARMISS_BASE_PATH", "/content/drive/MyDrive/Near_Miss_Scuola")

# Nome del CSV con le risposte del modulo Google Forms (Near Miss)
NOME_FILE_CSV_DEFAULT = "Segnalazione_Near_Miss-Sicurezza Scolastica_Risposte_del_modulo.csv"
# Percorso completo del CSV principale con le risposte
DEFAULT_CSV = f"{BASE_PATH}/xls/{NOME_FILE_CSV_DEFAULT}"

# Nome del CSV con il tracciato record (metadati delle colonne)
NOME_FILE_CSV_TRACCIATO_REC_DEFAULT = "tracciato_record.csv"
# Percorso completo del tracciato record
TRACCIATO_REC_DEFAULT_CSV = f"{BASE_PATH}/xls/{NOME_FILE_CSV_TRACCIATO_REC_DEFAULT}"

# =======================================================================
# ==========  SEZIONE: SCELTE ASSE X PER CSV PREDEFINITO  ===============
# =======================================================================
# --- Descrizione ---
# PREDEF_X_CHOICES:
#  - definisce quali colonne proporre come possibili assi X
#    quando si usa il CSV predefinito (Google Sheet “ufficiale”).
#  - La chiave (prima stringa) è il nome mostrato in UI,
#    il valore (seconda stringa) è il nome reale della colonna nel CSV.
# X_BLACKLIST_GENERIC:
#  - elenco di colonne "rumorose" che NON vanno proposte come asse X
#    nel caso di CSV generici (non standard).
# =======================================================================

PREDEF_X_CHOICES = OrderedDict([
    # "Etichetta visibile"        , "Nome reale della colonna nel CSV"
    ("📍 Regione",               "📍 Regione"),
    ("🏫 Luogo/Ambiente",        "🏫 Luogo/Ambiente dell'evento Near Miss"),
    ("⚠️ Possibili cause",       "⚠️ Possibili cause"),
    ("⚠️ Possibili conseguenze", "⚠️ Possibili conseguenze"),
])

# Colonne da escludere come asse X nel CSV generico
# (il match viene fatto in modo case-insensitive nel codice)
X_BLACKLIST_GENERIC = [
    "Informazioni cronologiche",
    "📝 NOTE",
    "📅 Data dell'evento Near Miss",
    "🕒 Ora indicativa dell'evento Near Miss",
]

# =======================================================================
# =================  SEZIONE: DEFAULT PLOT / DASHBOARD  =================
# =======================================================================
# --- Descrizione ---
# Qui si configurano i comportamenti di default dei grafici:
#  - per il CSV predefinito (Google Sheet “ufficiale”)
#  - per i CSV generici caricati dall’utente
# =======================================================================

# ----- CSV predefinito / Google Sheet -----

# Se True: per il CSV predefinito l’asse Y è sempre "conteggio righe"
Y_COUNT_ONLY_PREDEF: bool = True

# Se True: abilita il doppio asse Y per il CSV predefinito
# (Qui è disattivato: la dashboard ufficiale è mono-asse)
ENABLE_DUAL_Y_PREDEF: bool = False

# ----- CSV generico -----

# Se True: abilita il doppio asse Y per i CSV generici
ENABLE_DUAL_Y_GENERIC: bool = True

# Valore iniziale del checkbox “doppio asse Y” per il CSV generico
# False = l’utente deve abilitarlo manualmente
DEFAULT_DUAL_Y_GENERIC: bool = False

# Se True: permette “somma/media” come aggregazione su colonne numeriche vere
# (rimane comunque disponibile il conteggio righe come default)
Y_AGG_OPTIONS_GENERIC: bool = True

# =======================================================================
# =======  SEZIONE: RICONOSCIMENTO COLONNE NUMERICHE (GENERIC)  =========
# =======================================================================
# --- Descrizione ---
# Questi parametri servono a riconoscere quali colonne
# possono essere usate come metriche numeriche nei CSV generici.
# =======================================================================

# Minimo numero di valori non nulli per considerare una colonna “numerica utile”
NUMERIC_MIN_NON_NULL: int = 3

# Minimo numero di valori distinti richiesti (evita colonne con valore unico)
NUMERIC_MIN_UNIQUE: int = 2

# Colonne da escludere dalla valutazione numerica (case-insensitive)
NUMERIC_BLACKLIST: list[str] = ["note", "informazioni cronologiche"]

# =======================================================================
# =======  SEZIONE: DEBUG / PREFERENZE UI (CSV GENERICO)  ===============
# =======================================================================
# --- Descrizione ---
# Flag di debug per mostrare, in sidebar, informazioni aggiuntive
# sulle colonne numeriche del CSV generico.
# =======================================================================

# Se True: mostra nella sidebar la lista delle possibili metriche numeriche
DEBUG_NUMERIC_COLS_GENERIC: bool = True

# Se True: l’expander di debug numerico è aperto di default
DEBUG_NUMERIC_COLS_GENERIC_EXPANDED: bool = False

# Ordine di preferenza per la scelta automatica dell’asse X
# quando si carica un CSV generico.
GENERIC_X_DEFAULT_PREFERENCES: list[str] = [
    "📍 Regione",
    "🏫 Luogo/Ambiente dell'evento Near Miss",
    "⚠️ Possibili conseguenze",
    "⚠️ Possibili cause",
]

# =======================================================================
# =====================  SEZIONE: EXPOSE / TUNNEL  ======================
# =======================================================================
# --- Descrizione ---
# Configura il tipo di “tunnel” da usare per esporre Streamlit:
#  - "ngrok"     → usa Ngrok
#  - "cloudflare"→ usa Cloudflare Tunnel (Cloudflared)
# Lo stesso valore viene usato nella cella di AVVIO 2 in Colab.
# =======================================================================

# Provider di tunnel: "ngrok" oppure "cloudflare"
TUNNEL_PROVIDER: str = "cloudflare"

# Porta sulla quale Streamlit avvia la WebApp
STREAMLIT_PORT: int = 8501

# =======================================================================
# =======================  SEZIONE: RAG E LLM  ==========================
# =======================================================================
# --- Descrizione generale ---
# Qui si definiscono:
#  - modelli LLM (chat) per Google e OpenAI
#  - modelli di embedding testuale per il RAG
#  - opzioni di scansione dei file RAG (pattern e profondità)
#  - directory per rag_data, vectorstore, immagini e CSV RAG
# =======================================================================

# ------------------  Modelli LLM Google  ------------------
# GOOGLE_LLM_MODEL:
#  - modello principale usato per la chat con provider "Google"
#  - il prefisso "models/" è richiesto dalle API Gemini v1
#    (es. "models/gemini-2.5-flash" è uno dei modelli consigliati)
GOOGLE_LLM_MODEL = "models/gemini-2.5-flash"

# Modello di embedding testuale per Google
# "text-embedding-004" è l’ultimo embedding generico suggerito da Google.
GOOGLE_EMBEDDING_MODEL = "text-embedding-004"

# ------------------  Modelli LLM OpenAI  ------------------
# Modello chat (LLM) principale per OpenAI
OPENAI_LLM_MODEL = "gpt-4o-mini"

# Modello di embedding principale per OpenAI
# "text-embedding-3-small" offre un ottimo rapporto costo/prestazioni.
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ------------------  Glob pattern per file RAG  ------------------
# RAG_GLOB_PATTERNS:
#  - se None → usa il set di default (pdf, txt, csv, docx, xls, xlsx, png, jpg, jpeg, tif, tiff)
#  - se lista di pattern → filtra i file indicizzabili (es. ["*.pdf","*.txt","*.csv"])
RAG_GLOB_PATTERNS = None

# ------------------  Profondità di scansione rag_data/  ---------------
# RAG_SCAN_MAX_DEPTH controlla QUANTO in profondità scandire l’albero di cartelle:
#   None → nessun limite: scandisce tutta la struttura (rag_data + sottocartelle di qualsiasi livello)
#   1    → solo i file in rag_data/*                (primo livello)
#   2    → include rag_data/*/*                    (secondo livello)
#   3    → include rag_data/*/*/*                  (terzo livello)
#   ecc.
RAG_SCAN_MAX_DEPTH = None

# ------------------  Percorsi relativi al RAG  -------------------------

# Cartella che contiene i documenti usati dal RAG (pdf, txt, csv, ecc.)
RAG_DATA_DIR = f"{BASE_PATH}/rag_data"

# Nome del CSV "corrente" usato dalla dashboard come base dati
RAG_CSV_FILE = "dashboard.csv"

# Nomi alternativi eventualmente accettati come CSV di dashboard
RAG_CSV_ALIASES = ["dashboard.csv", "current_dashboard.csv"]

# Cartella dove viene salvato fisicamente il CSV del RAG (dashboard)
RAG_CSV_DIR = f"{RAG_DATA_DIR}/csv"

# Cartella dove viene salvato il Vector Store FAISS (indice RAG)
VECTORSTORE_DIR = f"{BASE_PATH}/vectorstore"

# Cartella per salvare le immagini generate dai grafici
IMG_DIR = f"{BASE_PATH}/img"

# Crea le cartelle se non esistono (evita errori in fase di salvataggio)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RAG_CSV_DIR, exist_ok=True)

# FIX #11 (v06_03): nella v06_02 il programma non controllava mai se la
# cartella principale del progetto (BASE_PATH) esistesse davvero.
# Quando Google Drive si disconnetteva (sessione Colab lunga, timeout,
# riavvio del runtime), tutti i percorsi diventavano invalidi e il
# programma mostrava errori difficili da capire.
# Ora facciamo un controllo subito: se la cartella non c'è, stampiamo
# un avviso chiaro nella console, così l'utente sa subito cosa fare.
# Nota: non usiamo st.error() qui perché questo file viene caricato
# come modulo e Streamlit potrebbe non essere ancora pronto.
if not os.path.isdir(BASE_PATH):
    import warnings
    warnings.warn(
        f"\n⚠️ ATTENZIONE: la cartella del progetto non è stata trovata!\n"
        f"   Percorso cercato: {BASE_PATH}\n"
        f"   Possibili cause:\n"
        f"   - Google Drive non è montato (riesegui la cella di mount in Colab)\n"
        f"   - Il percorso è cambiato (imposta la variabile d'ambiente NEARMISS_BASE_PATH)\n"
        f"   - Stai eseguendo in locale su un computer dove la cartella ha un nome diverso\n",
        stacklevel=2
    )

# ------------------  Google Sheet “ufficiale”  -------------------------
# ⚠️ ATTENZIONE ⚠️
# l'ID e il GID a seguire devono essere accoppiati e riferiti allo stesso Google Sheet 
# altrimenti l'URL della condivisione che si va a formare nelle righe successive, non funziona 
 
# ID del Google Sheet che contiene i dati Near Miss (foglio condiviso)
# ID del Google Sheet della scuola è "1jpRzYv5YVTOIyMHUDZ3S2oh7mr39LVWwYlupntcDt6Y"
# ID del Google Sheet simulato con circa 200 record è "1XnibaC6Hgq6VxvOAU7OYZtMIkYRqTTozgfdU9_U_Eo0"
ID_DEL_TUO_FOGLIO = "1jpRzYv5YVTOIyMHUDZ3S2oh7mr39LVWwYlupntcDt6Y"

# GID del foglio specifico all’interno del Google Sheet (tab)
# GID del foglio presente nello Sheet della scuola è "58099920"
# GID del foglio presente nello Sheet della simulazione con circa 200 record è "1420572456"
NUMERO_GID = "58099920"

# ⚠️ ATTENZIONE ⚠️
# l'ID e il GID richiamati nella composizione dell'URL a seguire, devono essere accoppiati e riferiti allo stesso Google Sheet 
# altrimenti l'URL della condivisione che si va a formare nelle righe successive, non funziona 
# URL di export in CSV del Google Sheet (usato per scaricare i dati aggiornati)
GOOGLE_SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{ID_DEL_TUO_FOGLIO}/export?format=csv&gid={NUMERO_GID}"
)

# =======================================================================
# ===================  SEZIONE: NOME BOT E PROMPT  ======================
# =======================================================================
# --- Descrizione ---
# Qui si definisce:
#  - il nome pubblico del bot (EduSafeAI)
#  - il prompt di sistema predefinito usato per istruire l’LLM
# =======================================================================

# Nome del bot (mostrato in UI e usato nel prompt)
BOT_NAME = "EduSafeAI"

# Prompt di sistema predefinito.
# Viene passato al modello come "istruzioni" di base per il comportamento del bot.
DEFAULT_SYSTEM_PROMPT = f"""
Sei {BOT_NAME}, un assistente virtuale esperto in Sicurezza sul Lavoro e in Near Miss nelle scuole e negli ambienti frequentati dai ragazzi/alunni, anche extra scolastici, come ad esempio nell'ambiente domestico (casa).

Se ti chiedono chi sei, rispondi personalizzando questa frase: "sono {BOT_NAME}, un assistente virtuale esperto in Sicurezza sul Lavoro e in Near Miss nelle scuole e negli ambienti frequentati dai ragazzi/alunni, anche extra scolastici, come ad esempio nell'ambiente domestico (casa)."

Se ti chiedono chi ti ha creato, inventato o qualcosa di simile, rispondi, personalizzando queste informazioni: 
"sono {BOT_NAME} e derivo da un progetto di Ricerca, ideato in INAIL, da un gruppo eterogeno di professionalità, per studiare i Near Miss, ossia i quasi incidenti, che potrebbero verificarsi nelle scuole e negli ambienti frequentati dai ragazzi/alunni, anche extra scolastici, come ad esempio nell'ambiente domestico (casa)"

Rispondi sempre in maniera chiara, concisa e professionale, basandoti principalmente sui documenti presenti 
nella Knowledge Base (RAG).
Se la domanda è fuori contesto o non hai dati a sufficienza per formulare una ragionevole risposta, sii chiaro e trasparente, specificando gentilmente che non puoi rispondere.

Mantieni un tono amichevole e usa un linguaggio semplice.

Nella conversazione, saluta soltanto se la conversazione è appena iniziata, ovvero evita di farlo ad ogni risposta, salvoche questa non sia la prima. Dunque, a conversazione iniziata, limitati a continuare la conversazione in modo gentile e professionale: ad esempio, non ripetere sempre (e così in analogia) "Ciao! Sono EduSafeAI... ".

"""

# =======================================================================
# ==========  SEZIONE: REGISTRAZIONE DEI MODELLI NELLA UI  ==============
# =======================================================================
# --- Descrizione ---
# Dizionario LLM_MODELS:
#  - mappa i nomi mostrati nell’interfaccia (selectbox, ecc.)
#    a:
#       - provider ("openai", "google", "locale", None)
#       - modello di embedding
#       - modello di chat (LLM)
#  - viene usato in Utils_RAG_NearMiss_v06_03.py e Utils_NearMiss_v06_03.py
# =======================================================================

LLM_MODELS = {
    "NESSUN MODELLO": {
        "description": (
            "⚠️ Nessun modello selezionato. Il bot non funzionerà finché non ne scegli uno.\n"
            "(OpenAI con text-embedding-3-small – Google Gemini con text-embedding-004 – Locale: non attivo in Cloud)"
        ),
        "provider": None,
        "embedding_model": None,
        "chat_model": None,
    },
    "OpenAI": {
        "description": (
            "✅ OpenAI – embedding: text-embedding-3-small; "
            "chat (LLM): gpt-4o-mini. Richiede OPENAI_API_KEY."
        ),
        "provider": "openai",
        "embedding_model": OPENAI_EMBEDDING_MODEL,  # Modello EMBEDDING predefinito per OpenAI
        "chat_model": OPENAI_LLM_MODEL,             # Modello LLM predefinito per OpenAI
    },
    "Google": {
        "description": (
            "✅ Google – embedding: text-embedding-004; "
            "chat (LLM): gemini-2.5-flash. Richiede credenziali Google."
        ),
        "provider": "google",
        "embedding_model": GOOGLE_EMBEDDING_MODEL,  # Modello EMBEDDING predefinito per Google
        "chat_model": GOOGLE_LLM_MODEL,             # Modello LLM predefinito per Google
    },
    "Locale": {
        "description": (
            "✅ Locale – es. sentence-transformers + LLM locale "
            "(ℹ️ DISABILITATO in Colab perché in Cloud; esempio commentato nel codice)."
        ),
        "provider": "locale",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "chat_model": "local-model",
    },
}

# =======================================================================
# ====================  SEZIONE: API KEY DA AMBIENTE  ===================
# =======================================================================
# --- Descrizione ---
# Qui NON si scrivono le chiavi in chiaro.
# Si leggono dalle variabili d'ambiente impostate in Colab (o altrove):
#   - OPENAI_API_KEY
#   - GOOGLE_API_KEY
# In questo modo:
#  - si evita di versionare chiavi sensibili
#  - si può cambiare chiave senza toccare i sorgenti .py
# =======================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
