# ============================================================
# Utils_RAG_NearMiss_v06_03.py
# ============================================================
# Modulo di utilità per la gestione del RAG (Retrieval-Augmented
# Generation) nel progetto Near Miss Scuola – EduSafeBot.
#
# Cosa fa questo file:
# - Gestisce il caricamento dei documenti (rag_data/…)
# - Divide i testi in chunk e crea gli embedding
# - Costruisce e carica l’indice vettoriale FAISS
# - Mantiene manifest e statistiche (n_file, n_chunks)
# - Fornisce la funzione principale rag_answer(...) usata dal bot
# - Gestisce la sincronizzazione del CSV (dashboard) con il RAG
#
# Tutto è pensato in modo didattico:
# - Funzioni separate per ogni responsabilità
# - Commenti riga per riga o per blocchi logici
# - Comportamento “robusto” in caso di errori (messaggi chiari)
# ============================================================

from __future__ import annotations  # consente annotazioni future (es. |, tuple[Any, int, int], ecc.)

# ============================================================
# ===============   IMPORT PRINCIPALI   ======================
# ============================================================

import os, json, glob, hashlib, requests   # os/path per file, json per serializzare, glob per pattern, hashlib (non usato molto), requests per HTTP diretto
import pandas as pd                        # gestione tabellare (CSV → DataFrame e viceversa)
import streamlit as st                     # UI web per la dashboard

import logging                             # logging per diagnostica
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
except Exception:
    # Fallback estremo
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = MockLogger()


from typing import List, Tuple, Dict, Optional, Any  # type hints
from pathlib import Path                             # gestione path in modo cross-platform
from statistics import mean                          # media aritmetica (per similarità)

import openai                                        # solo per intercettare errori tipici (model_not_found, ecc.)
import google.generativeai as genai                  # SDK Google Generative AI (Gemini)
from openai import OpenAI                            # client ufficiale OpenAI (non usato ovunque ma utile in debugging)

# ============================================================
# ===============   PARAMETRI DI BASE RAG   ==================
# ============================================================
# Modello di embedding HuggingFace usato di default per FAISS.
# È più leggero e adatto a CPU rispetto a modelli più pesanti.

EMBED_MODEL_HF = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Dimensione “teorica” del testo per chunk e sovrapposizione.
# chunk_size: quanti caratteri proviamo a mettere in ogni chunk
# chunk_overlap: quanto i chunk “si sovrappongono” per non tagliare concetti
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# ============================================================
# ===============  HELPER CHAT DIRETTI (LLM)  =================
# ============================================================
# Queste funzioni parlano direttamente con OpenAI / Gemini
# SENZA passare da LangChain, così evitiamo il problema 'proxies'
# dei wrapper e abbiamo pieno controllo sugli errori.

# ==============================
# --- OpenAI chat diretto ---
# ==============================
def _oai_chat(prompt: str, model: str) -> str:
    """
    Esegue una chiamata HTTP diretta alla API Chat Completions di OpenAI.
    - Usa OPENAI_API_KEY dall'ambiente
    - Evita l’uso di wrapper che inseriscono 'proxies'
    - Ritorna solo il testo della risposta
    """
    # Leggo la chiave dalle variabili d’ambiente
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Se manca la chiave, sollevo un errore esplicito
        raise RuntimeError("OPENAI_API_KEY mancante.")

    # Endpoint ufficiale per le chat
    url = "https://api.openai.com/v1/chat/completions"

    # Header con autenticazione Bearer
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Corpo della richiesta: modello, messaggi, temperatura, max_tokens
    payload = {
        "model": (model or "gpt-4o-mini"),
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    # Forzo l’assenza di proxy per evitare problemi in ambienti “pilotati”
    no_proxy = {"http": None, "https": None}

    # Chiamata HTTP
    r = requests.post(url, headers=headers, json=payload, proxies=no_proxy, timeout=60)
    r.raise_for_status()  # solleva eccezione se status != 200

    # Parsing della risposta JSON
    data = r.json()
    # Estraggo il testo della prima scelta (se presente)
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


# ==============================
# --- Google Gemini chat ---
# ==============================
def _gemini_chat(prompt: str, model: str) -> str:
    """
    Esegue una chiamata via SDK ufficiale Google Generative AI.
    - Usa GOOGLE_API_KEY
    - Il nome modello DEVE avere prefisso 'models/' (es. models/gemini-2.5-flash)
    """
    # Import locale per non appesantire l'avvio
    import google.generativeai as genai

    # Chiave API
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY mancante.")

    # Normalizzo il nome modello: se manca 'models/' lo aggiungo
    m = (model or "models/gemini-2.5-flash").strip()
    if not m.startswith("models/"):
        m = "models/" + m

    # Configuro il client
    genai.configure(api_key=api_key)

    # Creo il modello e invio il prompt
    g = genai.GenerativeModel(m)
    resp = g.generate_content(prompt)

    # Estraggo il testo in modo robusto
    txt = getattr(resp, "text", None)
    if not txt and getattr(resp, "candidates", None):
        txt = resp.candidates[0].content.parts[0].text

    return (txt or "").strip()


# ============================================================
# ===============   CARICAMENTO FAISS (CACHE)  ===============
# ============================================================
# _load_faiss_store carica il vectorstore FAISS dal disco
# e lo cache-izza con Streamlit. Tutti i punti del codice
# che devono leggere dal RAG passano da qui.

@st.cache_resource(show_spinner=False)
def _load_faiss_store(
    vs_dir: str,
    provider_embed: Optional[str] = None,
    embedding_model: Optional[str] = None
):
    """
    Carica una volta il vectorstore FAISS (cache tra i rerun Streamlit).

    Parametri:
    - vs_dir: cartella dove risiede l’indice FAISS (index.faiss, index.pkl, ecc.)
    - provider_embed: provider embedding da usare (di solito 'hf')
    - embedding_model: modello di embedding (se None usa EMBED_MODEL_HF)

    Ritorna:
    - oggetto FAISS caricato (LangChain vectorstore)
    """
    try:
        from langchain_community.vectorstores import FAISS

        # Se non passo provider/model, uso il picker automatico per l’indice
        prov = (provider_embed or _pick_embed_provider_for_index()[0])
        mdl = (embedding_model or (EMBED_MODEL_HF if prov == "hf" else None))

        # Creo l’embedder coerente col provider scelto
        emb = _embedding_factory(prov, mdl)

        # Carico l’indice FAISS da disco
        return FAISS.load_local(vs_dir, emb, allow_dangerous_deserialization=True)

    except Exception as e:
        # Messaggio di warning in UI, poi rilancio l’eccezione
        st.warning(f"⚠️ Impossibile caricare l’indice FAISS ({e}); verrà ricostruito se richiesto.")
        raise


# ============================================================
# ===============   PICKER EMBEDDING PER INDICE  =============
# ============================================================
# Centralizziamo la scelta del provider di embedding per l'indice.
# Per semplicità / prestazioni, usiamo sempre HuggingFace (HF)
# per costruire FAISS in questo progetto.

def _pick_embed_provider_for_index():
    """
    Ritorna il provider e il modello di embedding da usare
    quando costruiamo o leggiamo l’indice FAISS.
    Qui fissiamo sempre HF + EMBED_MODEL_HF.
    """
    return "hf", EMBED_MODEL_HF


# ============================================================
# ===============   OCR IMMAGINI (helper)   ==================
# ============================================================
def _estrai_testo_da_immagine(img_path, lingua="ita"):
    """
    Estrae testo da un file immagine usando Tesseract OCR.
    - img_path: path dell’immagine
    - lingua: codice lingua (es. 'ita')
    Ritorna testo grezzo (stringa) o stringa vuota in caso di errore.
    """
    import pytesseract
    from PIL import Image

    try:
        img = Image.open(img_path)
        testo = pytesseract.image_to_string(img, lang=lingua)
        return testo.strip()
    except Exception:
        return ""


# ============================================================
# ===============   EMBEDDING FACTORY (HF / API)  ============
# ============================================================
# Questa factory restituisce l’oggetto "Embeddings" usato da FAISS.
# - HF (HuggingFace) è il default
# - OpenAI e Google sono usati solo se configurati
# - c’è sempre un fallback HF se qualcosa va storto

@st.cache_resource(show_spinner=False)
def _embedding_factory(provider: str, embedding_model: Optional[str]) -> Any:
    """
    Restituisce un oggetto *Embeddings* di LangChain adatto a FAISS.

    provider:
      - 'hf' / 'huggingface' / 'sentence-transformers' → HuggingFaceEmbeddings
      - 'openai' → OpenAIEmbeddings (se API key presente)
      - 'google' → GoogleGenerativeAIEmbeddings (se API key presente)

    embedding_model:
      - nome del modello di embedding (o default EMBED_MODEL_HF)

    In caso di problemi, viene sempre usato HF come fallback.
    """
    prov = (provider or "").lower().strip()
    model = (embedding_model or EMBED_MODEL_HF).strip()

    # Helper interno: crea sempre un embedding HF con import lazy
    def _hf(model_name: str):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)

    # Caso HuggingFace (default)
    if prov in ("", "hf", "huggingface", "sentence-transformers"):
        model = model or EMBED_MODEL_HF
        return _hf(model)

    # Caso OpenAI embeddings
    if prov == "openai":
        from langchain_openai import OpenAIEmbeddings
        from config import OPENAI_API_KEY

        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            # nessuna chiave → fallback HF
            return _hf(EMBED_MODEL_HF)

        try:
            mdl = model or "text-embedding-3-small"
            return OpenAIEmbeddings(model=mdl, api_key=api_key)
        except Exception:
            # qualunque errore → fallback HF
            return _hf(EMBED_MODEL_HF)

    # Caso Google embeddings
    if prov == "google":
        from config import GOOGLE_API_KEY

        api_key = os.getenv("GOOGLE_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            return _hf(EMBED_MODEL_HF)

        try:
            mdl = model or "models/text-embedding-004"
            # Prefisso automatico se manca
            if not mdl.startswith("models/"):
                mdl = f"models/{mdl}"
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=mdl, google_api_key=api_key)
        except Exception:
            return _hf(EMBED_MODEL_HF)

    # Default di sicurezza: HF
    return _hf(EMBED_MODEL_HF)


# ============================================================
# ===============   FUNZIONI GESTIONE MANIFEST  ==============
# ============================================================
def _rag_glob_files(rag_dir: str) -> list[str]:
    """
    Elenca TUTTI i file indicizzabili sotto rag_dir (inclusi sottocartelle),
    rispettando:
    - Profondità massima configurata (RAG_SCAN_MAX_DEPTH)
    - Eventuali pattern personalizzati (RAG_GLOB_PATTERNS)
    - Eventuali alias per CSV (RAG_CSV_ALIASES)
    """
    from pathlib import Path

    # Import dei parametri di configurazione dal file config
    try:
        from config import (RAG_CSV_ALIASES, RAG_GLOB_PATTERNS, RAG_SCAN_MAX_DEPTH)
    except Exception:
        # Valori di fallback se non definiti
        RAG_CSV_ALIASES = []
        RAG_GLOB_PATTERNS = None
        RAG_SCAN_MAX_DEPTH = None

    root = Path(rag_dir)

    # Se non sono stati specificati pattern, uso quelli di default
    if not RAG_GLOB_PATTERNS:
        patterns = [
            "*.pdf", "*.txt", "*.csv", "*.docx", "*.xls", "*.xlsx",
            "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
        ]
    else:
        patterns = list(RAG_GLOB_PATTERNS)

    out: list[str] = []

    # Ciclo file ricorsivo per ogni pattern
    for pat in patterns:
        for p in root.rglob(pat):
            if RAG_SCAN_MAX_DEPTH is not None:
                # Calcolo profondità relativa rispetto a rag_dir
                rel_parts = p.relative_to(root).parts
                # Esempio: depth=1 accetta "pdf/file.pdf" ma non "pdf/sub/file.pdf"
                if len(rel_parts) > RAG_SCAN_MAX_DEPTH + 1:  # +1 per includere il file stesso
                    continue
            out.append(str(p))

    # Se sono configurati alias per CSV, filtro i CSV che non matchano quei nomi
    if RAG_CSV_ALIASES:
        out = [
            f for f in out
            if not f.lower().endswith(".csv") or (os.path.basename(f) in RAG_CSV_ALIASES)
        ]

    # Ordino per consistenza
    return sorted(out)


def _save_manifest_and_stats(vectorstore_dir: str, files: list[str], n_chunks: int | None):
    """
    Salva due file JSON:
    - sources.json: elenco file effettivi indicizzabili
    - stats.json: numero file + numero chunk
    """
    os.makedirs(vectorstore_dir, exist_ok=True)

    # Salvo lista file
    with open(os.path.join(vectorstore_dir, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)

    # Salvo statistiche
    stats = {
        "n_files": len(files),
        "n_chunks": int(n_chunks or 0),
    }
    with open(os.path.join(vectorstore_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _purge_vectorstore(dirpath: str):
    """
    Elimina i file dell’indice FAISS (index.faiss, index.pkl, sources.json, stats.json)
    per consentire una ricostruzione completamente pulita.
    """
    for name in ("index.faiss", "index.pkl", "sources.json", "stats.json"):
        p = os.path.join(dirpath, name)
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            # Se qualcosa va storto, non blocco il flusso
            pass


# ============================================================
# ===============   COSTRUZIONE INDICE FAISS  =================
# ============================================================
def build_vectorstore(
    root_dir: str,
    vs_dir: str,
    provider_embed: str = "hf",
    embedding_model: Optional[str] = None
) -> tuple[Any, int, int]:
    """
    Costruisce un nuovo indice FAISS partendo dai documenti in root_dir.

    Passi:
    - Carica i documenti (PDF/TXT/CSV/DOCX/XLS/immagini con OCR)
    - Li spezza in chunk con _split_documents()
    - Calcola embedding con _embedding_factory()
    - Crea il vectorstore FAISS e lo salva su vs_dir
    - Salva manifest (sources.json) e stats (stats.json)

    Ritorna:
    - vs: oggetto FAISS
    - n_files: numero di file sorgente unici
    - n_chunks: numero di chunk indicizzati
    """
    n_files = 0
    n_chunks = 0

    from langchain_community.vectorstores import FAISS

    # Carico i documenti (lista di Document LangChain)
    docs = _load_documents(root_dir)

    if not docs:
        # Nessun documento trovato → salvo manifest/stats vuoti per coerenza UI
        os.makedirs(vs_dir, exist_ok=True)
        with open(os.path.join(vs_dir, "sources.json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(os.path.join(vs_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({"n_files": 0, "n_chunks": 0}, f, ensure_ascii=False, indent=2)

        logger.info("Indicizzazione completata: 0 file, 0 chunk.")
        return None, 0, 0

    # Split in chunk
    chunks = _split_documents(docs)

    # Embedding factory (HF / OpenAI / Google)
    embeddings = _embedding_factory(provider_embed, embedding_model)

    # Costruisco l’indice FAISS
    vs = FAISS.from_documents(chunks, embeddings)

    # Creo cartella vs_dir se non esiste
    os.makedirs(vs_dir, exist_ok=True)

    # Salvo indice FAISS su disco
    vs.save_local(vs_dir)

    # Manifest sorgenti (file unici)
    srcs = []
    for d in docs:
        srcs.append((d.metadata or {}).get("source", "sconosciuta"))
    # Deduplica mantenendo ordine
    srcs = list(dict.fromkeys(srcs))

    n_files = len(srcs)
    n_chunks = len(chunks)
    
    # Calcolo l'hash dello stato attuale dei file nel RAG
    state_hash = _calculate_rag_state_hash(root_dir)

    # Salvo sources.json
    with open(os.path.join(vs_dir, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(srcs, f, ensure_ascii=False, indent=2)

    # Salvo stats.json (includendo l'hash dello stato)
    with open(os.path.join(vs_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"n_files": n_files, "n_chunks": n_chunks, "state_hash": state_hash},
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Indicizzazione completata: {n_files} file, {n_chunks} chunk. Hash: {state_hash}")
    return vs, n_files, n_chunks


def load_or_build_vectorstore(
    root_dir: str,
    vs_dir: str,
    provider: str,
    embedding_model: str
) -> tuple[Any, int, int]:
    """
    Tenta di caricare un vectorstore FAISS da vs_dir.
    Se fallisce o non esiste, lo ricostruisce da root_dir.
    """
    from langchain_community.vectorstores import FAISS

    if Path(vs_dir).exists():
        try:
            # Carico da disco usando embedding coerente
            vs = _load_faiss_store(vs_dir, provider, embedding_model)
            return vs, 0, 0
        except Exception:
            # Se il load fallisce, passo alla rebuild
            pass

    # Rebuild completo
    return build_vectorstore(root_dir, vs_dir, provider, embedding_model)


def refresh_manifest_stats(rag_dir: str, vectorstore_dir: str) -> tuple[int, int]:
    """
    Aggiorna SEMPRE sources.json e stats.json allo stato reale:
    - Ricava la lista dei file attualmente presenti in rag_dir
    - Legge i chunk da stats.json (senza caricare FAISS/HF)
    - Aggiorna manifest/stats senza ricostruire l’indice
    """
    n_chunks = 0

    files_now = _rag_glob_files(rag_dir)

    try:
        stats_p = Path(vectorstore_dir) / "stats.json"
        if stats_p.exists():
            with open(stats_p, "r", encoding="utf-8") as f:
                n_chunks = int((json.load(f) or {}).get("n_chunks", 0) or 0)
    except Exception:
        n_chunks = 0

    _save_manifest_and_stats(vectorstore_dir, files_now, n_chunks)
    return len(files_now), n_chunks


def _save_manifest_and_stats(vectorstore_dir: str, files: list, n_chunks: int, root_dir: Optional[str] = None):
    """
    Salva sources.json e stats.json nella cartella dell’indice.
    Se root_dir è fornito, calcola anche l'hash dello stato attuale.
    """
    os.makedirs(vectorstore_dir, exist_ok=True)
    
    # Manifest sorgenti
    with open(os.path.join(vectorstore_dir, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)

    # Statistiche
    stats: dict[str, Any] = {"n_files": len(files), "n_chunks": n_chunks}
    if root_dir:
        stats["state_hash"] = _calculate_rag_state_hash(root_dir)

    with open(os.path.join(vectorstore_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def ensure_index_smart(
    rag_dir: str,
    vectorstore_dir: str,
    provider_embed: str = "hf",
    embedding_model: Optional[str] = None,
    force: bool = False
) -> tuple[int, int]:
    """
    Versione “smart” della gestione indice:
    - Confronta la lista dei file live in rag_dir (files_live)
      con quella salvata in sources.json (old)
    - Confronta l'hash dello stato (nomi, mtime, size) con quello in stats.json
    - Se i set differiscono, l'hash è diverso o force=True → rebuild completo
    - Altrimenti aggiorna solo manifest/stats senza ricostruire.

    Ritorna (n_files, n_chunks) dopo l’operazione.
    """
    files_live = _rag_glob_files(rag_dir)

    n_chunks = 0
    n_files = 0

    # Carico stato precedente
    try:
        with open(os.path.join(vectorstore_dir, "sources.json"), "r", encoding="utf-8") as f:
            old_sources = json.load(f)
    except Exception:
        old_sources = []

    try:
        with open(os.path.join(vectorstore_dir, "stats.json"), "r", encoding="utf-8") as f:
            old_stats = json.load(f)
            old_hash = old_stats.get("state_hash", "")
    except Exception:
        old_hash = ""

    # Calcolo hash attuale
    current_hash = _calculate_rag_state_hash(rag_dir)

    # Decisione: se force=True, ma l'hash è uguale, potremmo comunque voler saltare 
    # se la lentezza è un problema. Ma se force=True è esplicito, diamogli priorità 
    # a meno che l'hash non ci dia la certezza di invarianza.
    # Ottimizzazione: anche con force=True, se current_hash == old_hash, saltiamo il rebuild.
    
    needs_rebuild = False
    if set(files_live) != set(old_sources):
        needs_rebuild = True
        reason = "cambiamento lista file"
    elif current_hash != old_hash:
        needs_rebuild = True
        reason = "cambiamento contenuto file (hash)"
    elif force and not old_hash:
        # Se richiesto force e non abbiamo un hash precedente, rebuild per sicurezza
        needs_rebuild = True
        reason = "forzato (no hash precedente)"

    if needs_rebuild:
        _, n_files, n_chunks = build_vectorstore(
            rag_dir,
            vectorstore_dir,
            provider_embed=provider_embed,
            embedding_model=embedding_model,
        )
        # build_vectorstore già salva manifest e stats con l'hash corretto
        logger.info(f"Rebuild RAG completato [{reason}]: {n_files} file, {n_chunks} chunk.")
        return n_files, n_chunks

    # Altrimenti, tengo l’indice ma aggiorno comunque manifest/stats (refresh leggero)
    logger.info("RAG Index coerente (hash ok): nessun ricalcolo embedding necessario.")
    return refresh_manifest_stats(rag_dir, vectorstore_dir)


# ============================================================
# ===============   UTILITY HASHING STATO RAG   ==============
# ============================================================
def _calculate_rag_state_hash(root_dir: str) -> str:
    """
    Calcola un hash basato su:
    - elenco file indicizzabili
    - taglia di ogni file
    - data di ultima modifica di ogni file
    Questo garantisce che se un file viene sovrascritto con lo stesso nome
    (es. dashboard.csv), l'hash cambi e il RAG si accorga del bisogno di rebuild.
    """
    import hashlib
    files = _rag_glob_files(root_dir)
    files.sort()  # garantisco ordine deterministico
    
    m = hashlib.md5()
    for fpath in files:
        p = Path(fpath)
        if p.is_file():
            stat = p.stat()
            # Aggiungo all'hash: path, taglia, mtime
            info = f"{fpath}|{stat.st_size}|{stat.st_mtime}"
            m.update(info.encode("utf-8"))
            
    return m.hexdigest()


# ============================================================
# ===============   CARICAMENTO DOCUMENTI RAG   ==============
# ============================================================
def _collect_documents() -> List[Tuple[str, str]]:
    """
    Versione semplificata del caricamento documenti:
    - Raccoglie coppie (path_file, testo) per txt/md/csv
    - Non usata nella pipeline principale (che usa _load_documents),
      ma lasciata come esempio didattico.
    """
    from config import RAG_DATA_DIR

    docs = []

    # Pattern di file da raccogliere (solo testo/markdown/csv)
    patterns = [
        os.path.join(RAG_DATA_DIR, "**", "*.txt"),
        os.path.join(RAG_DATA_DIR, "**", "*.md"),
        os.path.join(RAG_DATA_DIR, "**", "*.csv"),
    ]

    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            try:
                if p.lower().endswith(".csv"):
                    # CSV → lo ricodifico come testo (stile CSV)
                    df = pd.read_csv(p)
                    txt = df.to_csv(index=False)
                else:
                    # File di testo “grezzo”
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()

                if txt.strip():
                    docs.append((p, txt))
            except Exception:
                # Se un file fallisce la lettura, non blocco tutto
                continue

    return docs


# ============================================================
# ===============   OCR IMMAGINI (PUBBLICO)   =================
# ============================================================
def estrai_testo_da_immagine(img_path: str, lingua: str = "ita") -> str:
    """
    Wrapper pubblico per l’OCR:
    - img_path: path dell’immagine
    - lingua: codice lingua Tesseract (default 'ita')
    """
    import pytesseract
    from PIL import Image

    try:
        img = Image.open(img_path)
        testo = pytesseract.image_to_string(img, lang=lingua)
        return testo.strip()
    except Exception as e:
        print(f"[OCR] Errore durante la lettura di {img_path}: {e}")
        return ""


# ============================================================
# ===============   LOADER RICORSIVO COMPLETO   ===============
# ============================================================
def _load_documents(root_dir: str) -> list:
    """
    Carica i documenti indicizzabili seguendo ESATTAMENTE la lista prodotta da _rag_glob_files.
    Questo garantisce che l’esclusività dei CSV (RAG_CSV_ALIASES) sia rispettata.
    """
    files_to_load = _rag_glob_files(root_dir)
    docs = []

    # Per ogni tipo di file uso il loader appropriato
    for fpath in files_to_load:
        p = Path(fpath)
        ext = p.suffix.lower()

        try:
            # --- PDF ---
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                docs.extend(PyPDFLoader(str(p)).load())

            # --- TXT ---
            elif ext == ".txt":
                from langchain_community.document_loaders import TextLoader
                docs.extend(TextLoader(str(p), encoding="utf-8").load())

            # --- CSV ---
            elif ext == ".csv":
                from langchain_community.document_loaders import CSVLoader
                # Nota: loads rows as separate documents
                docs.extend(CSVLoader(str(p)).load())

            # --- DOCX ---
            elif ext == ".docx":
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                docs.extend(UnstructuredWordDocumentLoader(str(p)).load())

            # --- XLS / XLSX ---
            elif ext in [".xls", ".xlsx"]:
                from langchain_community.document_loaders import UnstructuredExcelLoader
                docs.extend(UnstructuredExcelLoader(str(p)).load())

            # --- Immagini (OCR) ---
            elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                from langchain.schema import Document
                t = _estrai_testo_da_immagine(str(p), lingua="ita")
                if t:
                    docs.append(Document(page_content=t, metadata={"source": str(p)}))
        
        except Exception as e:
            logger.warning(f"Errore caricamento {p}: {e}")

    return docs


# ============================================================
# ===============   SPLIT DOCUMENTS IN CHUNK  =================
# ============================================================
def _split_documents(documents: list):
    """
    Suddivide i Document in chunk di testo usando RecursiveCharacterTextSplitter.
    - chunk_size: CHUNK_SIZE
    - chunk_overlap: CHUNK_OVERLAP
    - separators: separatori preferiti per tagliare (paragrafi, frasi, spazi)
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ".", " "],
    )
    return splitter.split_documents(documents)


# ============================================================
# ===============   ENSURE INDEX (BASE)        ===============
# ============================================================
def ensure_index(
    root_dir: str,
    vs_dir: str,
    provider_embed: str = "hf",
    embedding_model: Optional[str] = None
) -> Tuple[int, int]:
    """
    Versione “semplice” di ensure_index:
    - Se index.faiss esiste, non ricostruisce (al massimo aggiorna stats.json)
    - Se index.faiss non esiste, costruisce l’indice da zero.

    Ritorna (n_file_creati, n_chunks_creati) se viene ricostruito, altrimenti (0,0).
    """
    idx = Path(vs_dir) / "index.faiss"

    if idx.exists():
        # Indice già presente → controllo solo stats.json
        s = get_index_stats(vs_dir)
        if not s:
            # Se mancano le stats, provo a ricavarle dal docstore
            try:
                vs = _load_faiss_store(vs_dir, "hf", EMBED_MODEL_HF)
                chunks = list(getattr(vs.docstore, "_dict", {}).values())
                sources = list({(d.metadata or {}).get("source", "sconosciuta") for d in chunks})
                with open(os.path.join(vs_dir, "stats.json"), "w", encoding="utf-8") as f:
                    json.dump({"n_files": len(sources), "n_chunks": len(chunks)}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return 0, 0

    # Se index.faiss non esiste, costruisco l’indice
    _, nf, nc = build_vectorstore(root_dir, vs_dir, provider_embed, embedding_model)
    return nf, nc


# ============================================================
# ===============   STATISTICHE INDICE       =================
# ============================================================
def get_index_stats(vs_dir: str) -> dict:
    """
    Ritorna le statistiche dell’indice (n_files, n_chunks) leggendo stats.json.
    Se non esiste, prova a ricavarle dal docstore.
    """
    p = Path(vs_dir) / "stats.json"

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback: calcolo diretto dal vectorstore (se riesco a caricarlo)
    try:
        vs = _load_faiss_store(vs_dir, "hf", EMBED_MODEL_HF)
        chunks = list(getattr(vs.docstore, "_dict", {}).values())
        sources = list({(d.metadata or {}).get("source", "sconosciuta") for d in chunks})
        return {"n_files": len(sources), "n_chunks": len(chunks)}
    except Exception:
        return {}


def get_index_sources(vs_dir: str) -> list[str]:
    """
    Ritorna la lista dei file sorgenti che compongono l’indice FAISS.
    - Prima prova a leggere sources.json
    - In fallback li ricava dal docstore, e poi salva sources.json
    """
    p = Path(vs_dir) / "sources.json"

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback: ricavo i sorgenti dal docstore e salvo sources.json
    try:
        vs = _load_faiss_store(vs_dir, "hf", EMBED_MODEL_HF)
        srcs = []
        for d in getattr(vs.docstore, "_dict", {}).values():
            srcs.append((d.metadata or {}).get("source", "sconosciuta"))
        # Dedup mantenendo ordine
        srcs = list(dict.fromkeys(srcs))

        os.makedirs(vs_dir, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(srcs, f, ensure_ascii=False, indent=2)

        return srcs
    except Exception:
        return []


def _structured_answer_from_dashboard_csv(query: str) -> Optional[Tuple[str, Dict]]:
    """
    Risposte deterministiche su dashboard.csv per domande tabellari semplici.
    Evita errori di conteggio del LLM su query tipo:
    - "Quanti near miss ... in Puglia?"
    - "In quali ambienti avvengono più spesso ... in Puglia?"
    """
    q = (query or "").strip()
    q_low = q.lower()

    if not q:
        return None

    is_count_q = ("quanti" in q_low) and ("near miss" in q_low)
    is_env_freq_q = ("ambient" in q_low) and (("frequent" in q_low) or ("più" in q_low) or ("piu" in q_low))

    if not (is_count_q or is_env_freq_q):
        return None

    from config import RAG_CSV_DIR, RAG_CSV_FILE

    csv_path = os.path.join(RAG_CSV_DIR, RAG_CSV_FILE)
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Colonne principali (ricerca robusta per nome)
    col_reg = next((c for c in df.columns if "regione" in str(c).lower()), None)
    col_env = next((c for c in df.columns if "luogo/ambiente" in str(c).lower() or "ambiente" in str(c).lower()), None)

    if not col_reg:
        return None

    # Regione richiesta: match sulle regioni presenti nel CSV
    region = None
    try:
        regioni = [str(x).strip() for x in df[col_reg].dropna().unique().tolist() if str(x).strip()]
        for r in regioni:
            if r.lower() in q_low:
                region = r
                break
    except Exception:
        region = None

    # Se non trovo regione esplicita, non forzo risposta deterministica
    if not region:
        return None

    dfr = df[df[col_reg].astype(str).str.strip().str.lower() == region.lower()].copy()

    if is_count_q:
        n = int(len(dfr))
        ans = f"Sono stati registrati {n} near miss accaduti in {region}."
        meta = {
            "chunks": max(1, n),
            "avg_score": 1.0,
            "avg_sim_01": 1.0,
            "files": [csv_path],
            "llm_model": "dashboard-analytics",
        }
        return ans, meta

    if is_env_freq_q and col_env:
        vc = dfr[col_env].astype(str).str.strip()
        vc = vc[vc != ""]
        counts = vc.value_counts()
        if counts.empty:
            ans = f"Non ho dati sugli ambienti per i near miss registrati in {region}."
        else:
            top = int(counts.iloc[0])
            ambienti_top = [str(i) for i, v in counts.items() if int(v) == top]
            if len(ambienti_top) == 1:
                ans = f"In {region}, l'ambiente con più near miss è {ambienti_top[0]} ({top} segnalazioni)."
            else:
                elenco = ", ".join(ambienti_top)
                ans = f"In {region}, gli ambienti con più near miss sono: {elenco} ({top} segnalazioni ciascuno)."

        meta = {
            "chunks": int(min(max(len(counts), 1), 12)),
            "avg_score": 1.0,
            "avg_sim_01": 1.0,
            "files": [csv_path],
            "llm_model": "dashboard-analytics",
        }
        return ans, meta

    return None


# ============================================================
# ===============   rag_answer (CUORE RAG)     ===============
# ============================================================
def rag_answer(query: str, provider: str, embedding_model: str, k: int = 4) -> Tuple[str, Dict]:
    """
    Funzione principale per il RAG.

    Passi:
    1. Carica l’indice FAISS con _load_faiss_store(...)
    2. Esegue similarity_search_with_score(query, k)
    3. Calcola statistiche (similarità, file coinvolti)
    4. Costruisce un prompt unificato (system + contesto + domanda)
    5. Chiama il modello LLM corrispondente al provider scelto
       - OpenAI → _oai_chat(...)
       - Google → _gemini_chat(...)
       - Locale → messaggio placeholder
    6. Ritorna:
       - answer: testo della risposta
       - meta: dizionario con info su chunk, score, file, modello LLM usato
    """
    n_chunks = 0

    from config import (
        RAG_DATA_DIR,
        VECTORSTORE_DIR,
        OPENAI_API_KEY,
        GOOGLE_API_KEY,
        DEFAULT_SYSTEM_PROMPT,
        LLM_MODELS,
        OPENAI_LLM_MODEL,
        GOOGLE_LLM_MODEL,
    )

    # Prima provo una risposta strutturata (deterministica) su dashboard.csv
    structured = _structured_answer_from_dashboard_csv(query)
    if structured is not None:
        return structured

    # Scelta embedder per l’indice (HF)
    prov_e, model_e = _pick_embed_provider_for_index()
    embeddings = _embedding_factory(prov_e, model_e)

    if embeddings is None:
        return (
            "Errore creazione embedding: impossibile proseguire.",
            {"chunks": 0, "avg_score": 0.0, "files": []},
        )

    # Carico il vectorstore FAISS
    # (Blocco rimosso in v06_03 - Fix #7: era ridondante e meno robusto del successivo)

    # --- Recupero/creazione indice (robusto) ---
    # In questa versione non facciamo rebuild automatico, solo load.
    try:
        vs = _load_faiss_store(VECTORSTORE_DIR, prov_e, model_e)
    except Exception as e:
        msg = (
            "ℹ️ L’indice RAG non è ancora pronto. Apri l’expander "
            "“📚 Indice RAG (FAISS)” e premi:\n"
            "• “💾 Sincronizza CSV → RAG” (se vuoi allineare il CSV attuale)\n"
            "• “🧱 Costruisci/Aggiorna indice” (per creare l’indice)"
        )
        return msg, {"chunks": 0, "avg_score": 0.0, "files": [], "error": str(e), "sources": []}

    # --- Retrieval robusto ---
    try:
        docs_scores = vs.similarity_search_with_score(query, k=k)
    except Exception:
        docs_scores = []

    # Calcolo media punteggio “raw” (cosine distance o simile)
    avg_score = float(mean([s for _, s in docs_scores])) if docs_scores else 0.0

    # Elenco file di origine (dedup-ordinato)
    files = [(doc.metadata or {}).get("source", "sconosciuta") for doc, _ in docs_scores]
    files = sorted(set(files))

    # Contesto concatenato
    context = "\n\n".join([doc.page_content for doc, _ in docs_scores]) if docs_scores else ""

    # --- Normalizzazione punteggi in [0..1] ---
    def _to_01(raw):
        """
        Converte un punteggio raw in [0..1].
        - Se già in [0..1] → lo lascia così (similarità)
        - Se >= 0 → lo interpreta come distanza (tipo 1/(1+x))
        """
        try:
            x = float(raw)
        except Exception:
            return 0.0
        if 0.0 <= x <= 1.0:
            return x
        if x >= 0.0:
            return 1.0 / (1.0 + x)
        return max(0.0, min(1.0, x))

    sims_01 = [_to_01(score) for _, score in docs_scores]
    avg_sim_01 = (sum(sims_01) / len(sims_01)) if sims_01 else 0.0

    if not docs_scores:
        # Nessun contesto utile dal RAG
        msg = (
            "Nei documenti RAG non ho trovato contenuti chiaramente rilevanti "
            "per la tua domanda. Prova a riformulare oppure aggiorna l’indice."
        )
        return msg, {"chunks": 0, "avg_score": 0.0, "files": []}

    # === Generazione risposta con LLM ===
    answer = ""
    prov = (provider or "").lower().strip()

    # Prompt unificato per tutti i provider
    prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Contesto RAG (top-k):\n{context}\n\n"
        f"Domanda dell'utente:\n{query}"
    )

    try:
        if prov == "openai":
            # Modello OpenAI da config (o default)
            chat_model = (
                LLM_MODELS.get("OpenAI", {}).get("chat_model")
                or OPENAI_LLM_MODEL
                or "gpt-4o-mini"
            )
            # Chiamata diretta OpenAI
            answer = _oai_chat(prompt, chat_model)
            model_used = chat_model

        elif prov == "google":
            # Modello Google da config (o default)
            g_model = (
                LLM_MODELS.get("Google", {}).get("chat_model")
                or GOOGLE_LLM_MODEL
                or "models/gemini-2.5-flash"
            )
            # Chiamata diretta Gemini
            answer = _gemini_chat(prompt, g_model)
            model_used = g_model

        elif prov == "locale":
            # Placeholder per eventuali futuri LLM locali
            answer = (
                "Il provider 'Locale' non è ancora disponibile in questa build. "
                "Seleziona OpenAI o Google dalla sidebar."
            )
            model_used = "(locale)"

        else:
            # Provider non riconosciuto
            answer = "Provider LLM non supportato per la generazione."
            model_used = "(sconosciuto)"

    except Exception as e:
        # In caso di errore, mostro un messaggio leggibile
        answer = f"❌ Errore chiamata LLM {prov.capitalize()}: {e}"
        # Provo a ricavare il nome modello usato, se possibile
        try:
            model_used
        except NameError:
            model_used = (
                LLM_MODELS.get("Google", {}).get("chat_model")
                if prov.startswith("google")
                else (LLM_MODELS.get("OpenAI", {}).get("chat_model") or "(modello non specificato)")
            )

    # Badge modello usato
    prov_low = prov

    try:
        llm_model = model_used
    except Exception:
        try:
            llm_model = GOOGLE_LLM_MODEL if prov_low.startswith("google") else OPENAI_LLM_MODEL
        except Exception:
            llm_model = (provider or "(modello non specificato)")

    # Dizionario meta-info per la UI (Fonte RAG)
    return answer, {
        "chunks": len(docs_scores),
        "avg_score": float(avg_score),
        "avg_sim_01": float(avg_sim_01),
        "files": files,
        "llm_model": llm_model,
    }


# ============================================================
# ===============   SYNC CSV DASHBOARD → RAG   ===============
# ============================================================
def sincronizza_csv_con_rag(df: pd.DataFrame, file_name: str = "dashboard.csv"):
    """
    Salva il CSV corrente della dashboard nella cartella RAG_CSV_DIR
    (configurata in config) con il nome file_name.

    Parametri:
    - df: DataFrame che rappresenta i dati della dashboard
    - file_name: nome del file CSV da salvare (default 'dashboard.csv')

    Ritorna:
    - path completo del file salvato (stringa) se OK
    - None in caso di errore
    """
    from config import RAG_CSV_DIR

    try:
        # Creo la cartella di destinazione se non esiste
        os.makedirs(RAG_CSV_DIR, exist_ok=True)

        # Path completo del CSV
        out_path = os.path.join(RAG_CSV_DIR, file_name)

        # Salvo il DataFrame in CSV UTF-8
        df.to_csv(out_path, index=False, encoding="utf-8")

        return out_path
    except Exception as e:
        st.error(f"❌ Errore salvataggio CSV per RAG: {e}")
        return None
