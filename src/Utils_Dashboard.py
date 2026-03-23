"""
Utils_NearMiss_v06_03.py
======================================
Funzioni di supporto per la WebApp "Near Miss Scuola – EduSafeBot".

Contiene:
- helper per grafici (dashboard)
- gestione dashboard/tabella + filtri
- gestione grafici predefiniti e generici (anche con doppio asse Y)
- gestione chat RAG (stato, rendering, invio domande)
- pannello RAG (gestione indice FAISS + micro-uploader)
- caricamento dati (Google Sheet / CSV)
- selezione LLM in sidebar
- sincronizzazione CSV → RAG

Versione v06_03 (Code Review Phase 2):
- Fix #8: corretta la chiave del messaggio in _add_chat ("content" invece di "text")
- Aggiornati import per puntare ai file v06_03.
"""

# ==========================
# IMPORT PRINCIPALI
# ==========================
import io, os, re, csv, json, glob, hashlib
from typing import Tuple, Optional

import pandas as pd
import streamlit as st

# Librerie per grafici
import numpy as np
import matplotlib.pyplot as plt
import logging                             # logging per diagnostica
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
except Exception:
    # Fallback estremo per evitare NameError
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = MockLogger()


# =====================================================================
# SEZIONE 1 — HELPER GRAFICI (etichette, top N, torte, ecc.)
# =====================================================================

def _pulisci_etichetta(s: str) -> str:
    """
    Pulisce una stringa da emoji / simboli iniziali e spazi superflui.
    Utile per etichette di assi e categorie nei grafici.
    """
    if s is None:
        return ""
    txt = str(s)
    txt = re.sub(r"^[^\w]+", "", txt).strip()
    return txt


def _top_series(series: pd.Series, topN: int, add_altri: bool):
    """
    Restituisce:
      - labels: categorie (come stringhe)
      - values: conteggi per categoria
      - total : somma totale dei conteggi

    Se le categorie sono più di topN:
      - tiene le topN più frequenti
      - se add_altri=True, raggruppa il resto nella categoria "Altri".
    """
    counts = series.value_counts(dropna=False)
    if len(counts) <= topN:
        labels = list(counts.index.astype(str))
        values = list(counts.values)
        total = int(counts.sum())
        return labels, values, total

    top = counts.head(topN)
    rest = counts.iloc[topN:].sum()
    labels = list(top.index.astype(str))
    values = list(top.values)
    total = int(counts.sum())
    if add_altri:
        labels.append("Altri")
        values.append(int(rest))
    return labels, values, total


def pulisci_label(s: str) -> str:
    """
    Versione “pubblica” di _pulisci_etichetta:
    rimuove emoji/prefix tipo '📍 ' dalle etichette.
    """
    if not isinstance(s, str):
        return s
    return re.sub(r"^[^\w\d]+", "", s).strip()


def _numeric_cols_utili(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """
    Restituisce solo le colonne numeriche “utili” come metriche, filtrando:
    - tipo numerico
    - numero minimo di valori non nulli
    - numero minimo di valori distinti
    - blacklist (es. NOTE, info cronologiche)
    - eventuale set di colonne da escludere esplicitamente.
    """
    if exclude is None:
        exclude = set()

    from config_v06_03 import (
        NUMERIC_MIN_NON_NULL,
        NUMERIC_MIN_UNIQUE,
        NUMERIC_BLACKLIST,
    )
    bl = {str(x).strip().lower() for x in NUMERIC_BLACKLIST}
    out: list[str] = []

    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        nn = s.notna().sum()
        if nn < NUMERIC_MIN_NON_NULL:
            continue
        if s.nunique(dropna=True) < NUMERIC_MIN_UNIQUE:
            continue
        if str(c).strip().lower() in bl:
            continue
        out.append(c)
    return out


def _default_index(cols: list[str], prefs: list[str]) -> int:
    """
    Ritorna l’indice del primo elemento di `prefs` presente in `cols`.
    Se nessuno è presente, restituisce 0 (prima colonna).
    """
    try:
        for p in prefs:
            if p in cols:
                return cols.index(p)
    except Exception:
        pass
    return 0 if cols else 0


def _agg_topn_percent(df, col, top_n=5, include_other=True):
    """
    Aggrega i valori di df[col] calcolando le percentuali sul totale.

    Restituisce:
    - labels: categorie
    - sizes : conteggi (non le percentuali, che vengono calcolate nel grafico)
    - total : totale dei record considerati
    """
    if df is None or df.empty or col not in df.columns:
        return [], [], 0

    vc = df[col].astype("string").fillna("—").value_counts()
    total = int(vc.sum())
    if total == 0:
        return [], [], 0

    top = vc.head(top_n)
    labels = list(top.index)
    sizes = list(top.values)

    if include_other:
        altri = total - int(sum(sizes))
        if altri > 0:
            labels.append("Altri")
            sizes.append(altri)

    return labels, sizes, total


def _draw_pie_topn(labels, sizes, title="", ax=None):
    """
    Disegna un grafico a torta con etichette di percentuale
    (mostrate solo se >= 0.5% per evitare rumore visivo).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    total = sum(sizes)
    if total == 0:
        ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    def _autopct(pct):
        return f"{pct:.1f}%" if pct >= 0.5 else ""

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        counterclock=False,
        autopct=_autopct,
        pctdistance=0.85,
        wedgeprops=dict(edgecolor="white"),
    )

    # Calcolo di un angolo medio per etichetta categoria
    angles = np.cumsum([0] + [w.theta2 - w.theta1 for w in wedges])
    mid = (angles[:-1] + angles[1:]) / 2.0
    r = 1.08  # raggio per posizionare le etichette vicino al bordo
    for lbl, ang in zip(labels, mid):
        x = r * np.cos(np.deg2rad(ang))
        y = r * np.sin(np.deg2rad(ang))
        ax.text(x, y, lbl, ha="center", va="center")

    ax.set_title(title)
    ax.axis("equal")
    return fig, ax


# =====================================================================
# SEZIONE 2 — PERSONALIZZAZIONE ASSI E GRAFICI (predefinito/generico)
# =====================================================================

def _generic_candidate_x_cols(df, blacklist: list[str]) -> list[str]:
    """
    Restituisce le colonne candidate per l’asse X, escludendo
    quelle in blacklist, preservando l’ordine originale.
    """
    bl = [s.lower() for s in (blacklist or [])]
    cols = [c for c in df.columns if c.lower() not in bl]
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _aggregate_series(df, x_col: str, mode: str, y_col: str | None, agg: str = "Somma"):
    """
    Aggrega i dati per costruire la serie Y a partire da X.

    mode:
      - "Conteggio righe" → df.groupby(x_col).size()
      - "Valore colonna numerica" → df.groupby(x_col)[y_col].sum()/mean
    """
    import pandas as pd
    if not x_col:
        return pd.Series(dtype="float64")
    if mode == "Conteggio righe" or not y_col:
        return df.groupby(x_col).size()
    g = df.groupby(x_col)[y_col]
    return g.sum() if (agg == "Somma") else g.mean()


def scegli_assi_personalizzati(df: pd.DataFrame, tipo: str):
    """
    UI (sidebar) per la selezione degli assi X e Y con CSV generici.

    - Per la torta: si sceglie un solo campo (X) da visualizzare.
    - Per istogrammi / barre / linee:
      * checkbox “⚙️ Personalizza asse X e Y”
      * selectbox per X
      * selectbox per Y (escludendo X)
      * inversione X/Y se tipo == "Barre orizzontali"
      * controllo per evitare che X e Y siano uguali.
    """
    x_col, y_col = None, None

    if tipo == "Torta":
        colonne = df.columns.tolist()
        x_col = st.sidebar.selectbox(
            "Campo da visualizzare (Torta)", colonne, key="custom_torta"
        )
        return x_col, None

    elif tipo in ["Istogramma (conteggio)", "Barre orizzontali", "Linee"]:
        personalizza = st.sidebar.checkbox("⚙️ Personalizza asse X e Y")
        if personalizza:
            colonne = df.columns.tolist()
            asse_x = st.sidebar.selectbox(
                "Campo da visualizzare sull'asse X", colonne, key="custom_x"
            )
            # lista candidati Y escluso l'X
            candidati_y = [c for c in colonne if c != asse_x]
            asse_y = st.sidebar.selectbox(
                "Campo da visualizzare sull'asse Y", candidati_y, key="custom_y"
            )
            # inversione logica per le barre orizzontali
            if tipo == "Barre orizzontali":
                x_col, y_col = asse_y, asse_x
            else:
                x_col, y_col = asse_x, asse_y

    # Se l’utente non ha personalizzato → Nessun asse impostato qui
    if x_col is None and y_col is None:
        return None, None

    # Protezione: X e Y non possono coincidere
    if x_col == y_col:
        st.warning("⚠️ X e Y non possono essere lo stesso campo. Seleziona colonne diverse.")
        return None, None

    return x_col, y_col


# =====================================================================
# SEZIONE 3 — DASHBOARD (tabella + grafici)
# =====================================================================

def visualizza_dashboard(df: pd.DataFrame):
    """
    Disegna la dashboard:
      - messaggio con numero record
      - filtri dinamici in sidebar
      - tabella filtrata + download CSV
      - sezione grafici (predefiniti vs generici).
    """
    if df is None or df.empty:
        st.warning("⚠️ Nessun dato disponibile per la dashboard.")
        return

    # Spinner mostrato solo al primo rendering della dashboard
    if not st.session_state.get("nm_first_paint_done", False):
        ph = st.empty()
        with ph.container():
            with st.spinner("⏳ preparo la dashboard…"):
                import time
                time.sleep(0.9)
        ph.empty()
        st.session_state["nm_first_paint_done"] = True

    st.success(f"✅ Record caricati: {len(df)}")

    # ----------------------------
    # FILTRI DINAMICI (sidebar)
    # ----------------------------
    st.sidebar.markdown("---")
    df_filt = df.copy()

    # In dashboard, nascondo la colonna data “eccessiva” se presente
    if "📅 Data dell'evento Near Miss" in df_filt.columns:
        df_filt = df_filt.drop(columns=["📅 Data dell'evento Near Miss"])

    with st.sidebar.expander("🔎 Filtri sui dati", expanded=False):
        for col in list(df_filt.columns):
            # Filtri solo per colonne testuali (evito filtri su numeriche libere)
            if df_filt[col].dtype == "object" or str(df_filt[col].dtype).startswith(
                "string"
            ):
                valori = sorted(df_filt[col].dropna().unique().tolist())
                if valori and len(valori) < 50:
                    selezione = st.sidebar.multiselect(f"Filtra {col}", valori)
                    if selezione:
                        df_filt = df_filt[df_filt[col].isin(selezione)]

    # ----------------------------
    # TABELLA + DOWNLOAD
    # ----------------------------
    st.markdown("### 📋 Tabella dati (filtrata)")
    st.dataframe(df_filt, use_container_width=True)
    csv_bytes = df_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Scarica tabella filtrata (CSV)",
        csv_bytes,
        "tabella_filtrata.csv",
        "text/csv",
        key="btn_scarica_csv_filtrato",
    )

    # ----------------------------
    # CONTROLLI GRAFICI (sidebar)
    # ----------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Grafici")
    tipo = st.sidebar.selectbox(
        "Tipo di grafico",
        ["Istogramma (conteggio)", "Barre orizzontali", "Torta", "Linee"],
        index=0,
    )
    st.sidebar.caption(
        "Suggerimento: parti con *Istogramma (conteggio)* e poi personalizza gli assi."
    )

    # Distinzione tra tracciato “Near Miss predefinito” e file generico
    is_predef = is_csv_predef(df)

    # Piccolo riepilogo dati correnti
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Dati correnti")
    try:
        _r, _c = df_filt.shape
    except Exception:
        _r = len(df_filt) if df_filt is not None else 0
        _c = len(df_filt.columns) if hasattr(df_filt, "columns") else 0
    st.sidebar.caption(
        ("Tracciato **predefinito**" if is_predef else "File **generico**")
        + f" • **{_r}** righe, **{_c}** colonne"
    )

    try:
        n_righe, n_colonne = df_filt.shape
    except Exception:
        n_righe = len(df_filt) if df_filt is not None else 0
        n_colonne = len(df_filt.columns) if hasattr(df_filt, "columns") else 0

    # ------------------------------------------------
    # CASO 3.1 — TRACCIATO PREDEFINITO (Near Miss)
    # ------------------------------------------------
    if is_predef:
        from config_v06_03 import (
            PREDEF_X_CHOICES,
            Y_COUNT_ONLY_PREDEF,
            ENABLE_DUAL_Y_PREDEF,
        )

        st.info(
            f"📄 Tracciato **Near Miss** riconosciuto (predefinito/Google Sheet). "
            f"Dati correnti: **{n_righe} righe**, **{n_colonne} colonne**."
        )

        # Scelta asse X tra 4 campi noti (Regione, Ambiente, ecc.)
        st.sidebar.markdown("### ⚙️ Asse X (predefinito)")
        asse_x_label = st.sidebar.radio(
            "Campo sull'asse X",
            list(PREDEF_X_CHOICES.keys()),
            index=0,
            key="predef_x_choice",
        )
        x_col = PREDEF_X_CHOICES[asse_x_label]

        # In questa modalità l’asse Y è SEMPRE il conteggio delle segnalazioni
        y_col = "__conteggio__"
        y2_col = None

        st.sidebar.caption(
            "ℹ️ Nel tracciato predefinito la metrica Y è il **conteggio delle segnalazioni**."
        )

        if Y_COUNT_ONLY_PREDEF:
            y_col = "__conteggio__"
            st.caption(
                "ℹ️ In questo tracciato, l'asse **Y** resta fisso al *conteggio delle segnalazioni*."
            )

        if not ENABLE_DUAL_Y_PREDEF:
            y2_col = None

        # Se per qualche motivo la colonna scelta non esiste, cerco un’alternativa tra le predefinite
        if x_col not in df_filt.columns:
            candidati = [
                c for c in PREDEF_X_CHOICES.values() if c in df_filt.columns
            ]
            if candidati:
                x_col = candidati[0]

        # TopN / “Altri”
        st.sidebar.markdown("---")
        topN = st.sidebar.slider("Mostra Top N", 3, 25, 10, key="predef_topN")
        mostra_tutti = st.sidebar.checkbox(
            "Mostra tutti", value=False, key="predef_all"
        )
        if mostra_tutti:
            flag_altri = False
            topN = df_filt[x_col].dropna().nunique()
            st.sidebar.caption("Mostro tutte le categorie → 'Altri' disattivato")
        else:
            flag_altri = st.sidebar.checkbox(
                "Includi 'Altri' come somma dei campi rimanenti",
                value=False,
                key="predef_altri",
            )

        # Opzioni rotazione etichette
        if tipo == "Torta":
            rotate_x = False
            rotate_y = False
            rotate_pie = st.sidebar.checkbox(
                "Ruota etichette (torta) lungo il raggio",
                value=False,
                key="predef_rot_pie",
            )
            st.sidebar.caption(
                "ℹ️ Per il grafico a torta gli assi X/Y non si applicano."
            )
        else:
            rotate_x = st.sidebar.checkbox(
                "Ruota etichette X (45°)", value=False, key="predef_rot_x"
            )
            rotate_y = st.sidebar.checkbox(
                "Ruota etichette Y (45°)", value=False, key="predef_rot_y"
            )
            rotate_pie = False

        # Render effettivo dei grafici predefiniti
        _render_grafici_predef(
            df_filt,
            tipo,
            x_col,
            y_col,
            topN,
            flag_altri,
            rotate_x,
            rotate_y,
            rotate_pie=rotate_pie,
        )

    # ------------------------------------------------
    # CASO 3.2 — CSV GENERICO (qualsiasi tracciato)
    # ------------------------------------------------
    else:
        from config_v06_03 import (
            DEBUG_NUMERIC_COLS_GENERIC,
            DEBUG_NUMERIC_COLS_GENERIC_EXPANDED,
            X_BLACKLIST_GENERIC,
            GENERIC_X_DEFAULT_PREFERENCES,
            DEFAULT_DUAL_Y_GENERIC,
        )

        st.info(
            f"📄 File **generico** caricato: **{n_righe} righe**, **{n_colonne} colonne**."
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Personalizzazione assi")

        personalizza = st.sidebar.checkbox(
            "Personalizza asse X e Y", value=False, key="gen_customize"
        )

        # Candidati per l’asse X (escludendo blacklist)
        all_x = _generic_candidate_x_cols(df_filt, X_BLACKLIST_GENERIC)

        # X di default = prima preferenza disponibile
        x_default = None
        for pref in (GENERIC_X_DEFAULT_PREFERENCES or []):
            if pref in all_x:
                x_default = pref
                break
        if not x_default and all_x:
            x_default = all_x[0]

        # Inizializzo alcune variabili per chiarezza
        x_col = None
        y_mode = "Conteggio righe"
        y_col = None
        agg_y = "Somma"
        use_y2 = False
        y2_mode = "Conteggio righe"
        y2_col = None
        agg_y2 = "Somma"

        # Caso: l’utente NON personalizza → X di default, Y = conteggio
        if not personalizza:
            x_col = x_default
            if tipo == "Torta":
                use_y2 = False
        else:
            # Scelta asse X
            x_col = st.sidebar.selectbox(
                "Campo (asse X)",
                all_x,
                index=(all_x.index(x_default) if (x_default in all_x) else 0)
                if all_x
                else 0,
                key="gen_x",
            )

            # Se c’era Y o Y2 settati uguali a X, li “sgancio”
            for key, label in (("gen_y_numcol", "Y"), ("gen_y2_numcol", "Y2")):
                cur = st.session_state.get(key)
                if cur == x_col:
                    st.info(
                        f"ℹ️ La colonna **{cur}** non può coincidere con l’asse X. "
                        f"Seleziona un valore diverso per {label}."
                    )
                    st.session_state[key] = None

            # Gestione asse Y
            if tipo == "Torta":
                y_mode = "Conteggio righe"
                y_col = None
                use_y2 = False
            else:
                y_mode = st.sidebar.radio(
                    "Asse Y",
                    ["Conteggio righe", "Valore colonna numerica"],
                    index=0,
                    key="gen_y_mode",
                )

                num_cols_all = df_filt.select_dtypes(include=["number"]).columns.tolist()

                # Debug: mostra colonne numeriche rilevate e quelle “utili”
                if DEBUG_NUMERIC_COLS_GENERIC:
                    raw_numeric = [
                        c
                        for c in df_filt.columns
                        if pd.api.types.is_numeric_dtype(df_filt[c])
                    ]
                    with st.sidebar.expander(
                        "🛠️ Debug metriche numeriche (CSV generico)",
                        expanded=DEBUG_NUMERIC_COLS_GENERIC_EXPANDED,
                    ):
                        st.caption(
                            f"Colonne numeriche rilevate (raw): {len(raw_numeric)}"
                        )
                        st.write(raw_numeric if raw_numeric else "—")
                        st.caption(
                            f"Colonne numeriche utili (dopo filtri): {len(num_cols_all)}"
                        )
                        st.write(num_cols_all if num_cols_all else "—")

                # Evita di riutilizzare come Y colonne già usate in X (e Y2, se numerico)
                conflict_set_y = {x_col}
                if (
                    "gen_y2_mode" in st.session_state
                    and st.session_state["gen_y2_mode"] == "Valore colonna numerica"
                ):
                    sel_y2 = st.session_state.get("gen_y2_numcol")
                    if sel_y2:
                        conflict_set_y.add(sel_y2)

                num_cols_y = [c for c in num_cols_all if c not in conflict_set_y]

                if y_mode == "Valore colonna numerica":
                    if not num_cols_y:
                        st.info(
                            "ℹ️ Nessuna metrica numerica disponibile (già usata da X/Y2 o inesistente): "
                            "uso il conteggio righe."
                        )
                        y_mode = "Conteggio righe"
                        y_col = None
                        agg_y = "Somma"
                    else:
                        prev = st.session_state.get("gen_y_numcol")
                        if prev and prev not in num_cols_y:
                            st.info(
                                f"ℹ️ La colonna **{prev}** non è selezionabile per Y perché già usata su un altro asse. "
                                "Scelgo una nuova opzione."
                            )
                            st.session_state["gen_y_numcol"] = num_cols_y[0]
                        y_col = st.sidebar.selectbox(
                            "Colonna numerica per Y",
                            num_cols_y,
                            key="gen_y_numcol",
                        )
                        agg_y = st.sidebar.radio(
                            "Aggregazione Y",
                            ["Somma", "Media"],
                            index=0,
                            key="gen_y_agg",
                        )
                else:
                    y_col = None
                    agg_y = "Somma"
                    if y_mode != "Valore colonna numerica":
                        st.session_state.pop("gen_y_numcol", None)

                # Asse Y secondario opzionale (solo se non torta)
                use_y2 = st.sidebar.checkbox(
                    "Attiva asse Y secondario (Y2)",
                    value=DEFAULT_DUAL_Y_GENERIC,
                    key="gen_use_y2",
                )
                if use_y2:
                    y2_mode = st.sidebar.radio(
                        "Asse Y2",
                        ["Conteggio righe", "Valore colonna numerica"],
                        index=0,
                        key="gen_y2_mode",
                    )

                    if y2_mode == "Valore colonna numerica":
                        # Candidati Y2 = numeriche diverse da X (e da Y se numerico)
                        conflict_set_y2 = {x_col}
                        if (
                            y_mode == "Valore colonna numerica"
                            and st.session_state.get("gen_y_numcol")
                        ):
                            conflict_set_y2.add(st.session_state["gen_y_numcol"])

                        candidates_y2 = [
                            c for c in num_cols_all if c not in conflict_set_y2
                        ]

                        if not candidates_y2:
                            st.info(
                                "ℹ️ Nessuna metrica numerica disponibile per Y2 "
                                "(già usata da X/Y o inesistente): uso il conteggio righe."
                            )
                            y2_mode = "Conteggio righe"
                            y2_col = None
                            agg_y2 = "Somma"
                        else:
                            prev2 = st.session_state.get("gen_y2_numcol")
                            if prev2 and prev2 not in candidates_y2:
                                st.info(
                                    f"ℹ️ La colonna **{prev2}** non è selezionabile per Y2 perché già usata su un altro asse. "
                                    "Scelgo una nuova opzione."
                                )
                                st.session_state["gen_y2_numcol"] = candidates_y2[0]
                            y2_col = st.sidebar.selectbox(
                                "Colonna numerica per Y2",
                                candidates_y2,
                                key="gen_y2_numcol",
                            )
                            agg_y2 = st.sidebar.radio(
                                "Aggregazione Y2",
                                ["Somma", "Media"],
                                index=0,
                                key="gen_y2_agg",
                            )
                    else:
                        y2_col = None
                        agg_y2 = "Somma"

        # Opzioni di visualizzazione TopN / Altri / rotazioni
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️️ Opzioni di visualizzazione")

        topN = st.sidebar.slider("Mostra Top N", 3, 25, 10, key="gen_topN")
        mostra_tutti = st.sidebar.checkbox(
            "Mostra tutti", value=False, key="gen_mostra_tutti"
        )
        if mostra_tutti:
            flag_altri = False
            base = x_col or (df_filt.columns[0] if len(df_filt.columns) else None)
            if base:
                topN = len(df_filt[base].astype(str).dropna().unique())
            st.sidebar.caption("Mostro tutte le categorie → 'Altri' disattivato")
        else:
            flag_altri = st.sidebar.checkbox(
                "Includi 'Altri' come somma dei campi rimanenti",
                value=False,
                key="gen_altri",
            )

        if tipo == "Torta":
            rotate_x = False
            rotate_y = False
            rotate_pie = st.sidebar.checkbox(
                "Ruota etichette (torta) lungo il raggio",
                value=False,
                key="gen_rot_pie",
            )
            st.sidebar.caption(
                "ℹ️ Per il grafico a torta gli assi X/Y non si applicano."
            )
        else:
            rotate_x = st.sidebar.checkbox(
                "Ruota etichette X (45°)", value=False, key="gen_rot"
            )
            rotate_y = st.sidebar.checkbox(
                "Ruota etichette Y (45°)", value=False, key="gen_rot_y"
            )
            rotate_pie = False

        # Disegno grafico (generico)
        if tipo == "Torta":
            if not x_col:
                st.info(
                    "ℹ️ Seleziona un campo valido per l'asse X per visualizzare il grafico."
                )
            else:
                _render_grafici_generico(
                    df_filt,
                    tipo,
                    x_col,
                    "__conteggio__",
                    topN,
                    flag_altri,
                    rotate_x=rotate_x,
                    rotate_y=rotate_y,
                    y2_col=None,
                    agg_mode="somma",
                    rotate_pie=rotate_pie,
                )
        else:
            if not x_col:
                st.info(
                    "ℹ️ Seleziona X (e Y se non vuoi il conteggio) per visualizzare il grafico."
                )
            else:
                # Aggregazione usata solo se Y è numerico
                agg_mode_lower = "somma"
                if personalizza and y_col and y_col != "__conteggio__":
                    agg_mode_lower = "somma" if agg_y == "Somma" else "media"

                _render_grafici_generico(
                    df_filt,
                    tipo,
                    x_col,
                    y_col,
                    topN,
                    flag_altri,
                    rotate_x=rotate_x,
                    rotate_y=rotate_y,
                    y2_col=y2_col,
                    agg_mode=agg_mode_lower,
                )


# ---------------------------------------------------------------------
# RENDER GRAFICI — TRACCIATO PREDEFINITO (Near Miss)
# ---------------------------------------------------------------------
def _render_grafici_predef(
    df: pd.DataFrame,
    tipo: str,
    x_col: str,
    y_col: str,
    topN: int,
    flag_altri: bool,
    rotate_x: bool = False,
    rotate_y: bool = False,
    rotate_pie: bool = False,
):
    """
    Grafici per CSV predefinito / Google Sheet.

    - y_col == '__conteggio__' → conteggio eventi (Near Miss).
    - (in futuro) si può prevedere una y numerica con sum/mean.
    """
    if df is None or df.empty or not x_col:
        st.info(
            "ℹ️ Seleziona un campo valido per l'asse X per visualizzare il grafico."
        )
        return

    x_lab = _pulisci_etichetta(x_col)
    y_lab = (
        "Numero di Segnalazioni"
        if y_col == "__conteggio__"
        else _pulisci_etichetta(y_col)
    )

    # ----- GRAFICO A TORTA -----
    if tipo == "Torta":
        labels, values, total = _top_series(df[x_col], topN, flag_altri)
        if total == 0 or sum(values) == 0:
            st.info("ℹ️ Nessun dato da visualizzare.")
            return

        # Percentuali sul totale, con correzione di arrotondamento
        denom = float(total)
        pcts_raw = [(100.0 * v / denom) for v in values]

        pcts_rounded = []
        if labels and labels[-1] == "Altri":
            acc = 0.0
            for i, pr in enumerate(pcts_raw):
                if i < len(pcts_raw) - 1:
                    v = round(pr, 1)
                    pcts_rounded.append(v)
                    acc += v
                else:
                    v = round(max(0.0, 100.0 - acc), 1)
                    pcts_rounded.append(v)
        else:
            pcts_rounded = [round(pr, 1) for pr in pcts_raw]

        lbls = [
            f"{_pulisci_etichetta(l)} – {p:.1f}%"
            for l, p in zip(labels, pcts_rounded)
        ]

        fig, ax = plt.subplots()
        if rotate_pie:
            wedges = ax.pie(values, labels=None, startangle=90)[0]
            r = 0.95  # etichette poco dentro il bordo
            for w, lbl in zip(wedges, lbls):
                ang = 0.5 * (w.theta1 + w.theta2)
                x = r * np.cos(np.deg2rad(ang))
                y = r * np.sin(np.deg2rad(ang))
                ax.text(
                    x,
                    y,
                    lbl,
                    ha="center",
                    va="center",
                    rotation=ang,
                    rotation_mode="anchor",
                )
        else:
            ax.pie(values, labels=lbls, startangle=90)

        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        return

    # ----- BARRE / LINEE -----
    if y_col == "__conteggio__":
        labels, values, _ = _top_series(df[x_col], topN, flag_altri)
        x_vals = [_pulisci_etichetta(l) for l in labels]
        y_vals = values
    else:
        # (non usata in UI al momento, tenuta per completezza)
        agg = df.groupby(x_col)[y_col].count().sort_values(ascending=False)
        if topN and len(agg) > topN:
            agg = agg.head(topN)
        x_vals = [_pulisci_etichetta(x) for x in agg.index.tolist()]
        y_vals = agg.values.tolist()

    fig, ax = plt.subplots()

    if tipo == "Barre orizzontali":
        ax.barh(x_vals, y_vals)
        ax.set_xlabel(y_lab)
        ax.set_ylabel(x_lab)
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

    elif tipo == "Linee":
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

    else:  # Istogramma (conteggio)
        ax.bar(x_vals, y_vals)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

    st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------
# RENDER GRAFICI — CSV GENERICO
# ---------------------------------------------------------------------
def _render_grafici_generico(
    df: pd.DataFrame,
    tipo: str,
    x_col: str,
    y_col: Optional[str],
    topN: int,
    flag_altri: bool,
    rotate_x: bool = False,
    rotate_y: bool = False,
    y2_col: Optional[str] = None,
    agg_mode: str = "somma",
    rotate_pie: bool = False,
):
    """
    Grafici per CSV generico.

    - Se y_col è None o '__conteggio__' → si conta il numero di righe per categoria X.
    - Se y_col è numerico               → si aggrega per X (somma o media).
    - y2_col (numerico)                 → solo per grafico a linee (asse Y secondario).
    """
    if df is None or df.empty or not x_col:
        st.info(
            "ℹ️ Seleziona un campo valido per l'asse X per visualizzare il grafico."
        )
        return

    serie_x_raw = df[x_col]
    serie_x = serie_x_raw.astype(str).fillna("(vuoto)")

    # ----------------------
    # COSTRUZIONE X/Y
    # ----------------------
    x_vals: list = []
    y_vals: list = []
    y_lab = "Conteggio"

    if y_col in (None, "__conteggio__"):
        # Conteggio categorie X con TopN + “Altri”
        conteggi = serie_x.value_counts()
        if topN < len(conteggi):
            head = conteggi.head(topN)
            if flag_altri:
                altri = int(conteggi.iloc[topN:].sum())
                head = pd.concat([head, pd.Series({"Altri": altri})])
        else:
            head = conteggi

        x_vals = [_pulisci_etichetta(x) for x in head.index.tolist()]
        y_vals = head.values.tolist()
        y_lab = "Conteggio"

    else:
        # Y numerico → somma o media per X
        tmp = df[[x_col, y_col]].copy()
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
        tmp = tmp.dropna(subset=[y_col])

        if agg_mode == "media":
            agg = tmp.groupby(x_col, dropna=False)[y_col].mean().sort_values(
                ascending=False
            )
            y_lab = f"{_pulisci_etichetta(y_col)} (media)"
        else:
            agg = tmp.groupby(x_col, dropna=False)[y_col].sum().sort_values(
                ascending=False
            )
            y_lab = f"{_pulisci_etichetta(y_col)} (somma)"

        if topN < len(agg):
            head = agg.head(topN)
            if flag_altri:
                resto = agg.iloc[topN:]
                altri_val = resto.mean() if agg_mode == "media" else resto.sum()
                head = pd.concat([head, pd.Series({"Altri": altri_val})])
        else:
            head = agg

        x_vals = [_pulisci_etichetta(x) for x in head.index.tolist()]
        y_vals = head.values.tolist()
        y_lab = _pulisci_etichetta(y_col)

    fig, ax = plt.subplots()

    # ----- TORTA (generico) -----
    if tipo == "Torta":
        conteggi = serie_x.value_counts()
        if topN < len(conteggi):
            head = conteggi.head(topN)
            if flag_altri:
                altri = int(conteggi.iloc[topN:].sum())
                head = pd.concat([head, pd.Series({"Altri": altri})])
        else:
            head = conteggi

        labels = [_pulisci_etichetta(x) for x in head.index.tolist()]
        values = head.values.tolist()
        total = float(sum(values))
        if total == 0 or sum(values) == 0:
            st.info("ℹ️ Nessun dato da visualizzare.")
            return

        denom = float(total)
        pcts_raw = [(100.0 * v / denom) for v in values]

        pcts_rounded = []
        if labels and labels[-1] == "Altri":
            acc = 0.0
            for i, pr in enumerate(pcts_raw):
                if i < len(pcts_raw) - 1:
                    v = round(pr, 1)
                    pcts_rounded.append(v)
                    acc += v
                else:
                    v = round(max(0.0, 100.0 - acc), 1)
                    pcts_rounded.append(v)
        else:
            pcts_rounded = [round(pr, 1) for pr in pcts_raw]

        lbls = [
            f"{_pulisci_etichetta(l)} – {p:.1f}%"
            for l, p in zip(labels, pcts_rounded)
        ]

        if rotate_pie:
            wedges = ax.pie(values, labels=None, startangle=90)[0]
            r = 0.95
            for w, lbl in zip(wedges, lbls):
                ang = 0.5 * (w.theta1 + w.theta2)
                x = r * np.cos(np.deg2rad(ang))
                y = r * np.sin(np.deg2rad(ang))
                ax.text(
                    x,
                    y,
                    lbl,
                    ha="center",
                    va="center",
                    rotation=ang,
                    rotation_mode="anchor",
                )
        else:
            ax.pie(values, labels=lbls, startangle=90)

        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        return

    # ----- BARRE ORIZZONTALI -----
    elif tipo == "Barre orizzontali":
        ax.barh(x_vals, y_vals)
        ax.set_xlabel(y_lab)
        ax.set_ylabel(_pulisci_etichetta(x_col))
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

    # ----- LINEE (con possibile Y2) -----
    elif tipo == "Linee":
        ax.plot(x_vals, y_vals, marker="o", label=y_lab)
        ax.set_xlabel(_pulisci_etichetta(x_col))
        ax.set_ylabel(y_lab)
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

        # Asse Y secondario (se richiesto)
        if y2_col:
            if y2_col not in df.columns:
                st.warning("⚠️ Campo per asse Y secondario non valido.")
            else:
                tmp2 = df[[x_col, y2_col]].copy()
                tmp2[y2_col] = pd.to_numeric(tmp2[y2_col], errors="coerce")
                tmp2 = tmp2.dropna(subset=[y2_col])

                if agg_mode == "media":
                    agg2 = tmp2.groupby(x_col, dropna=False)[y2_col].mean()
                    y2_lab = f"{_pulisci_etichetta(y2_col)} (media)"
                else:
                    agg2 = tmp2.groupby(x_col, dropna=False)[y2_col].sum()
                    y2_lab = f"{_pulisci_etichetta(y2_col)} (somma)"

                agg2 = agg2.reindex(
                    pd.Index([s for s in serie_x_raw.dropna().unique()]),
                    fill_value=np.nan,
                )

                map2 = {str(k): v for k, v in agg2.items()}
                y2_vals = [
                    map2.get(orig, np.nan)
                    for orig in df[x_col].astype(str).unique()
                    if _pulisci_etichetta(orig) in x_vals
                ]
                if len(y2_vals) != len(x_vals):
                    idx_map = {
                        _pulisci_etichetta(k): v for k, v in agg2.items()
                    }
                    y2_vals = [idx_map.get(x, np.nan) for x in x_vals]

                ax2 = ax.twinx()
                ax2.plot(x_vals, y2_vals, linestyle="--", marker="o", label=y2_lab)
                ax2.set_ylabel(y2_lab)
                ax2.tick_params(axis="y", labelrotation=0, pad=10)
                ax2.set_ylabel(y2_lab, labelpad=12)

                # Legenda combinata
                _apply_rotation_y(ax2, rotate=rotate_y, angle=45)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                legend = ax.legend(
                    lines + lines2,
                    labels + labels2,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=2,
                    frameon=True,
                    framealpha=0.85,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.93))
                ax.margins(x=0.02)

    # ----- ISTOGRAMMA VERTICALE -----
    else:
        ax.bar(x_vals, y_vals)
        ax.set_xlabel(_pulisci_etichetta(x_col))
        ax.set_ylabel(y_lab)
        _apply_rotation(ax, rotate=rotate_x, angle=45)
        _apply_rotation_y(ax, rotate=rotate_y, angle=45)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ===================================================================================
# SEZIONE 4 — ROTAZIONE LABEL ASSI + GRAFICO CON DOPPIO ASSE Y (helper generico)
# ===================================================================================

def _apply_rotation(ax, rotate: bool = False, angle: int = 45):
    """Ruota le etichette dell’asse X se rotate=True."""
    if rotate:
        for t in ax.get_xticklabels():
            t.set_rotation(angle)
            t.set_ha("right")


def _apply_rotation_y(ax, rotate: bool = False, angle: int = 45):
    """Ruota le etichette dell’asse Y se rotate=True."""
    if not rotate or ax is None:
        return
    for lab in ax.get_yticklabels():
        lab.set_rotation(angle)
        lab.set_ha("right")
        lab.set_va("center")


def _plot_with_dual_axis(x, y1, kind="bar", label1="Conteggi", label2="% cumulata"):
    """
    Esempio di grafico con doppio asse Y (non usato nel flusso principale,
    ma utile come riferimento didattico).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    if kind == "bar":
        ax.bar(x, y1, label=label1)
    elif kind == "barh":
        ax.barh(x, y1, label=label1)
    elif kind == "line":
        ax.plot(x, y1, marker="o", label=label1)

    # Asse Y secondario con percentuale cumulata
    y2 = (pd.Series(y1).cumsum() / sum(y1) * 100).values
    ax2 = ax.twinx()
    ax2.plot(x, y2, marker="o", linestyle="--", label=label2)

    _apply_rotation(ax, st.sidebar.checkbox("Ruota etichette X a 45°", value=False))

    ax.set_ylabel(label1)
    ax2.set_ylabel(label2)
    ax2.set_ylim(0, 105)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    st.pyplot(fig, use_container_width=True)


# =====================================================================
# SEZIONE 5 — GESTIONE CHAT (stato + rendering)
# =====================================================================

def _init_state():
    """
    Inizializza lo stato di sessione per:
    - chat_log
    - provider LLM
    - gestione input/clear
    - scelta fonte dati (Google Sheet / CSV)
    - tracking fonte_corrente per sync CSV → RAG.

    Viene chiamata dal main prima di usare la chat o la dashboard.
    """
    # CSS per chat e bottoni (stile uniforme)
    st.markdown(
        """
    <style>
    /* margine verticale per box chat utente/bot */
    div[data-testid="stMarkdownContainer"] > div[style*="background:#eef"] { margin: .25rem 0; }
    div[data-testid="stMarkdownContainer"] > div[style*="background:#efe"] { margin: .25rem 0; }

    /* stile uniforme per tutti i bottoni (Streamlit button, download, submit) */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
      background-color: #002e5f !important;  /* blu INAIL */
      color: #ffffff !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 0.6rem 1rem !important;
      font-weight: 600 !important;
    }

    .buttons-row {
      display: flex;
      gap: 0.75rem;
      justify-content: center;
      align-items: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Stato chat
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    if "last_provider" not in st.session_state:
        st.session_state["last_provider"] = None

    if "_clear_input" not in st.session_state:
        st.session_state["_clear_input"] = False

    if "textarea_domanda" not in st.session_state:
        st.session_state["textarea_domanda"] = ""

    if "last_answer_files" not in st.session_state:
        st.session_state["last_answer_files"] = []

    # Stato per la sorgente dati (Selezione Esclusiva)
    if "tipo_sorgente" not in st.session_state:
        st.session_state["tipo_sorgente"] = "💾 CSV Locale (Default)"

    if "fonte_corrente" not in st.session_state:
        st.session_state["fonte_corrente"] = ""
        
    if "rag_needs_rebuild" not in st.session_state:
        st.session_state["rag_needs_rebuild"] = False

    if "last_dashboard_hash" not in st.session_state:
        st.session_state["last_dashboard_hash"] = ""


def _normalize_chat_log():
    """
    Normalizza il formato di st.session_state["chat_log"]:

    - converte eventuali tuple (role, content) in dict {"role":..., "content":...}
    - scarta formati anomali per evitare errori di rendering.
    """
    log = st.session_state.get("chat_log", [])
    changed = False
    norm = []
    for m in log:
        if isinstance(m, tuple) and len(m) == 2:
            norm.append({"role": m[0], "content": m[1]})
            changed = True
        elif isinstance(m, dict) and "role" in m and "content" in m:
            norm.append(m)
        else:
            continue
    if changed:
        st.session_state["chat_log"] = norm


def _render_chat_container() -> None:
    """
    Alias per compatibilità con versioni precedenti.
    Chiama semplicemente render_chat().
    """
    render_chat()


def _chat_as_txt():
    """
    Esporta la chat in un formato testuale lineare.
    Utile per il download della conversazione.
    """
    lines = []
    for msg in st.session_state.get("chat_log", []):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
        else:
            continue
        who = "Utente" if role == "user" else "Assistente"
        lines.append(f"{who}: {content}")
    return "\n\n".join(lines)


def _model_badge_from_meta(provider: str, meta: dict) -> str:
    """
    Restituisce una stringa compatta con il nome del provider e del modello
    (es. 'Google gemini-1.5-pro' oppure 'OpenAI gpt-4.1-mini').
    """
    prov = (provider or "").lower()
    prov_name = "Google" if prov.startswith("google") else "OpenAI"
    model = meta.get("llm_model") or meta.get("model") or "(modello non specificato)"
    return f"{prov_name} {model}"


def gestisci_chatbot(provider: str, embedding_model: str):
    """
    Gestisce tutta la sezione “Chatbot (RAG)” nella pagina principale:
    - visualizza il pannello RAG (📚 Documenti & Indice)
    - controlla se l’indice è vuoto
    - mostra la form per inviare domande
    - esegue la chiamata RAG (rag_answer)
    - aggiorna lo storico della chat.
    """
    # Gestione della pulizia input (dopo invio o clear)
    if st.session_state.get("_clear_input", False):
        st.session_state["_clear_input"] = False
        st.session_state.pop("textarea_domanda", None)

    _normalize_chat_log()

    # Garantisco struttura coerente del log
    st.session_state.setdefault("chat_log", [])
    st.session_state.setdefault("_clear_input", False)

    normalized = []
    for m in st.session_state.get("chat_log", []):
        if isinstance(m, dict):
            role = m.get("role", "assistant")
            content = m.get("content", m.get("text", ""))
            normalized.append({"role": role, "content": content})
        elif isinstance(m, tuple):
            role = m[0] if len(m) > 0 else "assistant"
            content = m[1] if len(m) > 1 else ""
            normalized.append({"role": role, "content": content})
        else:
            normalized.append({"role": "assistant", "content": str(m)})
    st.session_state["chat_log"] = normalized

    st.subheader("🤖 Chatbot (RAG)")

    # Pannello documenti/indice RAG
    pannello_rag(provider, embedding_model, suffix="_chat")

    # Avviso se indice RAG vuoto
    try:
        from config_v06_03 import VECTORSTORE_DIR
        from Utils_RAG_NearMiss_v06_03 import get_index_stats

        stats_now = get_index_stats(VECTORSTORE_DIR) or {}
    except Exception:
        stats_now = {}

    if int(stats_now.get("n_chunks", 0) or 0) == 0:
        st.warning(
            "ℹ️ L'indice RAG è attualmente vuoto (**0 chunk**). "
            "Apri il pannello **📚 Documenti & Indice (RAG)** qui sopra e usa "
            "“(Ri)costruisci indice” prima di fare richieste al Bot."
        )

    # -------------------------
    # FORM DI DOMANDA ALLA CHAT
    # -------------------------
    st.markdown("**Invia una domanda ai documenti**")
    with st.form("ask_form", clear_on_submit=False):
        default_value = ""
        if st.session_state.get("_clear_input"):
            default_value = ""
            st.session_state["_clear_input"] = False
        else:
            default_value = st.session_state.get("textarea_domanda", "")

        domanda = st.text_area(
            "Scrivi qui (la casella cresce e ha scroll oltre 4–5 righe):",
            value=default_value,
            key="textarea_domanda",
            height=140,
            placeholder="es. Quali sono le principali cause di Near Miss nei laboratori?",
        )

        # Riga di bottoni allineati
        st.markdown('<div class="buttons-row">', unsafe_allow_html=True)
        col_ask, col_clear = st.columns([1, 1])
        with col_ask:
            from config_v06_03 import BOT_NAME

            invia = st.form_submit_button(
                f"💬 Chiedi qualcosa a {BOT_NAME}",
                type="primary",
                use_container_width=True,
            )

        with col_clear:
            clear = st.form_submit_button(
                "🗑️ Svuota chat", use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # Gestione pulsante CLEAR
        # -------------------------
        if clear:
            if not (st.session_state.get("textarea_domanda") or "").strip():
                st.warning("L'area di input non contiene nulla da cancellare 🙂")
            else:
                st.session_state["chat_log"] = []
                st.session_state["last_answer_files"] = []
                st.session_state["_clear_input"] = True
                st.session_state["_clear_chat"] = True
                st.rerun()

        # -------------------------
        # Gestione pulsante INVIA
        # -------------------------
        if invia:
            testo = (domanda or "").strip()
            if not testo:
                st.warning("Scrivi una domanda prima di inviare 🙂")
            elif not provider or provider == "NESSUN MODELLO":
                st.warning("⚠️ Seleziona un LLM in sidebar prima di inviare.")
            else:
                try:
                    from config_v06_03 import VECTORSTORE_DIR
                    from Utils_RAG_NearMiss_v06_03 import (
                        get_index_stats,
                        rag_answer,
                        ensure_index_smart,
                        EMBED_MODEL_HF,
                    )
                    
                    if st.session_state.get("rag_needs_rebuild", False):
                        with st.spinner("🔄 Aggiorno l'indice RAG prima di rispondere..."):
                            ensure_index_smart(
                                RAG_DATA_DIR,
                                VECTORSTORE_DIR,
                                provider_embed="hf",
                                embedding_model=EMBED_MODEL_HF,
                                force=True,
                            )
                        st.session_state["rag_needs_rebuild"] = False
                    
                    stats = get_index_stats(VECTORSTORE_DIR)
                    if (stats.get("n_chunks") or 0) == 0:
                        st.info(
                            "ℹ️ L’indice RAG è vuoto (0 chunk). "
                            "Puoi ricostruirlo dal pannello 📚 Documenti & Indice (RAG)."
                        )
                        return

                    # k più alto migliora la copertura dei record CSV (es. dashboard.csv)
                    # nelle domande aggregate per regione/ambiente.
                    answer, meta = rag_answer(testo, provider, embedding_model, k=12)
                    
                    # Append domanda utente
                    st.session_state["chat_log"].append(
                        {"role": "user", "content": testo}
                    )

                except Exception as e:
                    prov_name = (
                        "Google"
                        if str(provider).lower().startswith("google")
                        else "OpenAI"
                    )
                    st.error(f"❌ Errore chiamata LLM {prov_name}: {e}")
                    return

                # Costruzione “fonte RAG” con info su chunk, similarità, file e modello
                fonte = "__ **Fonte RAG** __\n"
                fonte += f"- Chunk rilevanti: {meta.get('chunks', 0)}\n"

                sim01 = meta.get("avg_sim_01", None)
                if sim01 is None:
                    sim_fallback = meta.get("avg_score", 0.0)
                    try:
                        sim_fallback = float(sim_fallback)
                    except Exception:
                        sim_fallback = 0.0
                    fonte += f"- Similarità media: {sim_fallback:.2f}\n"
                else:
                    fonte += (
                        f"- Similarità media: {float(sim01):.2f} (0=bassa, 1=alta)\n"
                    )

                files = meta.get("files") or []
                if files:
                    fonte += f"- File coinvolti ({len(files)}):\n"
                    for f in files:
                        fonte += f"  • {os.path.basename(f)}\n"
                    st.session_state["last_answer_files"] = files
                else:
                    st.session_state["last_answer_files"] = []

                fonte += (
                    f"- Risposta data dal modello: {_model_badge_from_meta(provider, meta)}\n"
                )

                st.session_state["chat_log"].append(
                    {"role": "assistant", "content": answer + "\n\n" + fonte}
                )

                st.session_state["_clear_input"] = True
                st.success("✅ Domanda inviata al bot.")
                st.rerun()

    # -------------------------
    # STORICO + DOWNLOAD CHAT
    # -------------------------
    st.markdown("**Storico conversazione**")
    render_chat()
    st.download_button(
        "⬇️ Scarica chat (.txt)",
        data=_chat_as_txt(),
        file_name="conversazione.txt",
        mime="text/plain",
        use_container_width=True,
        key="btn_scarica_chat",
    )


def _add_chat(role: str, text: str):
    """
    Helper semplice per aggiungere un messaggio allo storico.
    
    Fix #8 (v06_03): Usiamo "content" invece di "text" per coerenza
    con il resto della gestione chat (`render_chat` si aspetta "content"
    o "text", ma lo standard interno è "content").
    """
    st.session_state.chat_log.append({"role": role, "content": text})


def render_chat() -> None:
    """
    Renderizza lo storico della chat.

    - Messaggi utente: allineati a destra, sfondo azzurrino.
    - Messaggi bot: allineati a sinistra, sfondo verdino.
    """
    logs = st.session_state.get("chat_log", [])
    if not logs:
        st.info("Nessun messaggio nella conversazione.")
        return

    for msg in logs:
        if isinstance(msg, dict):
            role = msg.get("role", "assistant")
            text = msg.get("content")
            if text is None:
                text = msg.get("text", "")
        elif isinstance(msg, tuple):
            role = msg[0] if len(msg) > 0 else "assistant"
            text = msg[1] if len(msg) > 1 else ""
        else:
            role, text = "assistant", str(msg)

        if role == "user":
            st.markdown(
                "<div style='text-align:right; background:#eef; padding:.45rem; "
                "border-radius:.45rem; margin:.22rem 0;'>"
                f"{text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:left; background:#efe; padding:.45rem; "
                "border-radius:.45rem; margin:.22rem 0;'>"
                f"{text}</div>",
                unsafe_allow_html=True,
            )


# =====================================================================
# SEZIONE 6 — SELEZIONE LLM IN SIDEBAR
# =====================================================================

def seleziona_llm_sidebar(LLM_MODELS: dict) -> Tuple[str, str]:
    """
    Mostra in sidebar:
    - selectbox per scegliere il provider LLM
    - descrizione sintetica del modello
    - stato corrente (LLM / Embedding)
    - stato API Key (OpenAI/Google).
    Restituisce:
      provider, embedding_model
    """
    st.sidebar.header("🧠 Modello LLM / Embedding")
    provider = st.sidebar.selectbox(
        "Seleziona il provider", list(LLM_MODELS.keys()), index=0
    )
    st.sidebar.caption(LLM_MODELS[provider]["description"])

    if st.session_state.get("last_provider") != provider:
        st.session_state["last_provider"] = provider
        st.session_state["_clear_input"] = True

    embedding_model = LLM_MODELS[provider]["embedding_model"]
    chat_model_val = LLM_MODELS[provider]["chat_model"]
    if hasattr(chat_model_val, "model_name"):
        chat_model_pretty = chat_model_val.model_name
    else:
        chat_model_pretty = str(chat_model_val)

    st.sidebar.markdown("### 📌 Stato corrente")
    if provider and provider != "NESSUN MODELLO":
        st.sidebar.success(
            f"✅ **LLM**: {provider} ({chat_model_pretty})\n"
            f"✅ **Embedding**: {provider} ({embedding_model})"
        )
    else:
        st.sidebar.warning(
            "⚠️ Nessun LLM/Embedding selezionato.\n"
            "Seleziona un modello dalla lista sopra per iniziare."
        )

    # Stato API key (parte centrale della pagina, non in sidebar)
    st.markdown("### 🔑 Stato API Key")
    if provider.lower() == "openai":
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.success("✅ API Key OpenAI caricata", icon="✅")
        else:
            st.error("❌ API Key OpenAI mancante", icon="🚫")
    elif provider.lower() == "google":
        if os.getenv("GOOGLE_API_KEY", "").strip():
            st.success("✅ API Key Google caricata", icon="✅")
        else:
            st.error("❌ API Key Google mancante", icon="🚫")
    elif provider in ("NESSUN MODELLO", "Locale"):
        st.error("❌ Seleziona un Modello LLM / Embedding 🧠", icon="🚫")

    st.markdown("---")
    return provider, embedding_model


# =====================================================================
# SEZIONE 7 — CARICAMENTO DATI (Google Sheet / CSV)
# =====================================================================

@st.cache_data(ttl=600)  # Cache per 10 minuti per non bombardare Google Sheets
def carica_da_google_sheet(url: str, timeout: int = 6) -> Optional[pd.DataFrame]:
    """
    Carica il CSV pubblico di un Google Sheet in modo robusto:

    - verifica che l’URL risponda (HTTP 200)
    - in caso di errore mostra un warning in Streamlit
    - restituisce None se non riesce a leggere i dati.
    """
    import urllib.request, urllib.error

    if not url:
        st.warning("⚠️ URL del Google Sheet mancante nel file di configurazione.")
        return None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                st.warning(f"⚠️ Il Google Sheet non è raggiungibile (HTTP {r.status}).")
                return None

        df = pd.read_csv(url)
        return df

    except urllib.error.URLError as e:
        st.warning(f"⚠️ Connessione a Google Sheet non riuscita ({e.reason}).")
        st.warning(f"⚠️ Impossibile leggere i dati dal Google Sheet: {url}")

    except pd.errors.EmptyDataError:
        st.warning(
            "⚠️ Il Google Sheet è vuoto o non contiene dati leggibili."
        )
        st.warning(f"⚠️ Impossibile leggere i dati dal Google Sheet: {url}")

    except Exception as e:
        st.warning(f"⚠️ Errore durante la lettura del Google Sheet: {str(e)}")
        st.warning(f"⚠️ Impossibile leggere i dati dal Google Sheet: {url}")

    return None


@st.cache_data(show_spinner=False)
def carica_dati(fonte: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Carica i dati in base alla fonte scelta:

    - fonte == "google": prova Google Sheet; se fallisce, passa al CSV locale.
    - fonte == "csv"   : usa direttamente il CSV locale.

    Restituisce (df, descrizione_fonte).
    """
    df, fonte_dati = None, ""
    if fonte == "google":
        from config_v06_03 import GOOGLE_SHEET_URL

        df = carica_da_google_sheet(GOOGLE_SHEET_URL)
        if df is not None:
            fonte_dati = "Google Sheet (tempo reale)"

    if df is None:
        try:
            from config_v06_03 import DEFAULT_CSV

            df = pd.read_csv(DEFAULT_CSV)
            fonte_dati = "CSV locale/predefinito"
        except FileNotFoundError:
            st.error("❌ Nessun CSV predefinito trovato.")
            return None, ""

    return df, fonte_dati

# Il timeout nella def è il tempo di attesa per il controllo del Google Sheet, rendendolo più stabile nel Cloud (Colab).
def google_sheet_available(timeout_sec: int = 5) -> Tuple[bool, str]:
    """
    Ritorna (True, "OK") se l’URL CSV pubblico del Google Sheet risponde (HTTP 200).
    Altrimenti ritorna (False, "motivo errore").
    Modificato v06_03.8.2: evita st.session_state nel thread (non sicuro).
    """
    import concurrent.futures
    import urllib.request, urllib.error

    def _probe():
        try:
            from config_v06_03 import GOOGLE_SHEET_URL
        except Exception as e:
            return False, f"Errore configurazione: {str(e)}"

        if not GOOGLE_SHEET_URL:
            return False, "URL Google Sheet non configurato."

        try:
            req = urllib.request.Request(
                GOOGLE_SHEET_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                if r.status == 200:
                    return True, "OK"
                else:
                    return False, f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Connessione fallita: {str(e)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_probe)
        try:
            res = fut.result(timeout=timeout_sec)
            return res  # res è (bool, str)
        except Exception as e:
            return False, f"Timeout o Errore: {str(e)}"


def read_google_with_timeout(timeout_sec: int = 10) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Legge il Google Sheet in un thread separato con timeout.
    
    Fix #6 (v06_03): Unificazione della funzione di lettura che prima
    era duplicata nel main script.
    """
    import concurrent.futures
    # Nota: carica_dati è già definita in questo file module-level.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        # Carica_dati("google") restituisce (df, fonte_str)
        fut = ex.submit(carica_dati, "google")
        try:
            return fut.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Google Sheet non ha risposto entro {timeout_sec}s")


# =====================================================================
# SEZIONE 8 — PANNELLO RAG (Documenti & Indice + Micro-uploader)
# =====================================================================

def pannello_rag(provider: str, embedding_model: str, suffix: str = ""):
    """
    Mostra il pannello RAG con statistiche e pulsanti di gestione indice:

    - Documenti & Indice (RAG): conteggio file e chunk, pulsanti:
        * 🔁 (Ri)costruisci indice
        * 🧹 Pulisci e ricostruisci
        * 🧭 Solo aggiorna manifest/stats
        * 📎 Scarica lista file indicizzabili (live)
    - Micro-uploader per aggiungere nuovi file in rag_data/ con instradamento
      automatico nelle sottocartelle (pdf/txt/csv/doc/xls/img).
    - Expander “Diagnostica (tecnica)” per visualizzare sources.json e stats.json.

    Il parametro suffix serve a rendere uniche le chiavi Streamlit se il pannello
    viene richiamato più volte nella stessa pagina.
    """
    # ⚡ Ottimizzazione performance (v06_03.9):
    # non eseguiamo più check/refresh automatico dell'indice qui.
    # L'allineamento indice avviene al cambio sorgente o via pulsanti manuali.

    # -------------------
    # EXPANDER PRINCIPALE
    # -------------------
    with st.expander("📚 Documenti & Indice (RAG)", expanded=False):
        from config_v06_03 import RAG_DATA_DIR, VECTORSTORE_DIR
        from Utils_RAG_NearMiss_v06_03 import (
            get_index_stats,
            build_vectorstore,
            EMBED_MODEL_HF,
            refresh_manifest_stats,
            _rag_glob_files,
            _save_manifest_and_stats,
            _purge_vectorstore,
        )

        st.caption(f"Percorso documenti RAG (informativo):\n\n{RAG_DATA_DIR}")

        # Statistiche correnti su file e chunk
        stats = get_index_stats(VECTORSTORE_DIR) or {}
        n_files = stats.get("n_files", 0)
        n_chunks = stats.get("n_chunks", 0)
        st.write(f"**Documenti indicizzati:** {n_files} • **Chunk:** {n_chunks}")
        
        if st.session_state.get("rag_needs_rebuild", False):
            st.warning("⚠️ Sorgente dati aggiornata: indice RAG da allineare (al prossimo invio chat o via pulsanti manuali).")


        # --- Colonna 1: (Ri)costruisci indice ---
        cols = st.columns(4)
        with cols[0]:
            if st.button(
                "🔁 (Ri)costruisci indice",
                key=f"btn_ricostruisci_rag{suffix}",
                help=(
                    "Forza una ricostruzione completa dell’indice FAISS dai file presenti in rag_data/ "
                    "(anche se non sono cambiati)."
                ),
            ):
                try:
                    _, n_files, n_chunks = build_vectorstore(
                        RAG_DATA_DIR,
                        VECTORSTORE_DIR,
                        provider_embed="hf",
                        embedding_model=EMBED_MODEL_HF,
                    )
                    _save_manifest_and_stats(
                        VECTORSTORE_DIR, _rag_glob_files(RAG_DATA_DIR), n_chunks
                    )
                    st.success(
                        f"✅ Indice (ri)costruito. File: {n_files} • Chunk: {n_chunks}"
                    )
                except Exception as e:
                    st.error(f"❌ Ricostruzione indice fallita: {e}")

        # --- Colonna 2: Pulisci e ricostruisci ---
        with cols[1]:
            if st.button(
                "🧹 Pulisci e ricostruisci",
                key=f"btn_pulisci_rag{suffix}",
                help=(
                    "Cancella l’indice FAISS esistente e lo ricrea da zero. "
                    "Usalo se sospetti corruzioni o mismatch persistenti."
                ),
            ):
                try:
                    _purge_vectorstore(VECTORSTORE_DIR)
                    files_now = _rag_glob_files(RAG_DATA_DIR)
                    _, n_files, n_chunks = build_vectorstore(
                        RAG_DATA_DIR,
                        VECTORSTORE_DIR,
                        provider_embed="hf",
                        embedding_model=EMBED_MODEL_HF,
                    )
                    _save_manifest_and_stats(
                        VECTORSTORE_DIR, _rag_glob_files(RAG_DATA_DIR), n_chunks
                    )
                    st.success(
                        f"✅ Pulizia + ricostruzione completate. File: {n_files} • Chunk: {n_chunks}"
                    )
                except Exception as e:
                    st.error(f"❌ Operazione fallita: {e}")

        # --- Colonna 3: Solo aggiorna manifest/stats ---
        with cols[2]:
            if st.button(
                "🧭 Solo aggiorna manifest/stats (senza rebuild)",
                key=f"btn_refresh_manifest{suffix}",
                help=(
                    "Aggiorna l’elenco file (manifest) e i conteggi, senza ricostruire l’indice. "
                    "Utile dopo aver aggiunto/eliminato file."
                ),
            ):
                n_files, n_chunks = refresh_manifest_stats(RAG_DATA_DIR, VECTORSTORE_DIR)
                st.success(
                    f"✅ Manifest aggiornato. File: {n_files} • Chunk: {n_chunks}"
                )

        # --- Colonna 4: Download lista file indicizzabili ---
        with cols[3]:
            try:
                srcs = _rag_glob_files(RAG_DATA_DIR)
            except Exception:
                srcs = []
            st.download_button(
                "📎 Scarica lista file indicizzabili (live)",
                help=(
                    "Scarica l’elenco dei file che al momento verrebbero considerati "
                    "per l’indicizzazione (scansione ricorsiva)."
                ),
                data=("\n".join(srcs)).encode("utf-8"),
                file_name="rag_sources.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"btn_dl_sources{suffix}",
            )

        # -------------------
        # MICRO-UPLOADER FILE
        # -------------------
        st.divider()
        st.subheader("➕ Aggiungi documenti a rag_data/")
        upl = st.file_uploader(
            "Trascina qui PDF/TXT/CSV/DOCX/XLS/XLSX/immagini",
            type=[
                "pdf",
                "txt",
                "csv",
                "docx",
                "xls",
                "xlsx",
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff",
            ],
            accept_multiple_files=True,
        )
        if upl:
            from pathlib import Path

            def _dst_subdir(ext: str) -> str:
                """
                Decide in quale sottocartella di rag_data/ salvare il file
                in base all’estensione.
                """
                ext = ext.lower()
                if ext in ("pdf",):
                    return "pdf"
                if ext in ("txt",):
                    return "txt"
                if ext in ("csv",):
                    return "csv"
                if ext in ("docx",):
                    return "doc"
                if ext in ("xls", "xlsx"):
                    return "xls"
                if ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                    return "img"
                return "_scarti"

            saved = 0
            base = Path(RAG_DATA_DIR)
            for f in upl:
                name = f.name
                ext = name.split(".")[-1].lower() if "." in name else ""
                sub = _dst_subdir(ext)
                if sub == "_scarti":
                    st.warning(f"Formato non supportato: {name}")
                    continue
                dst_dir = base / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                with open(dst_dir / name, "wb") as out:
                    out.write(f.read())
                saved += 1

            if saved:
                st.success(
                    f"✅ Caricati {saved} file. Ora aggiorna i conteggi o ricostruisci l’indice."
                )
                if st.button(
                    "🔄 Aggiorna manifest/stats adesso",
                    key=f"btn_refresh_after_upload{suffix}",
                ):
                    n_files, n_chunks = refresh_manifest_stats(
                        RAG_DATA_DIR, VECTORSTORE_DIR
                    )
                    st.info(
                        f"Manifest aggiornato. File: {n_files} • Chunk: {n_chunks}"
                    )

    # -------------------
    # EXPANDER DIAGNOSTICA
    # -------------------
    with st.expander("🛠️ Diagnostica (tecnica)", expanded=False):
        """
        Mostra:
        - sources.json → elenco file usati per costruire l’indice
        - stats.json   → conteggi chunk, versione, ecc.

        Utile per supporto tecnico quando qualcosa “non va”.
        """
        st.caption(
            "Se qualcosa non funziona, apri questo box e incolla eventuali errori nella chat di supporto."
        )
        import json as _json

        try:
            src_p = os.path.join(VECTORSTORE_DIR, "sources.json")
            with open(src_p, "r", encoding="utf-8") as f:
                st.json(_json.load(f), expanded=False)
        except Exception:
            st.write("sources.json non disponibile.")
        try:
            with open(os.path.join(VECTORSTORE_DIR, "stats.json"), "r", encoding="utf-8") as f:
                st.json(_json.load(f), expanded=False)
        except Exception:
            st.write("stats.json non disponibile.")


# =====================================================================
# SEZIONE 9 — RICONOSCIMENTO TRACCIATO / SYNC CSV → RAG
# =====================================================================

def get_tracciato_record_headers() -> list[str]:
    """
    Restituisce l’header atteso del CSV predefinito/realtime.

    - Se TRACCIATO_REC_DEFAULT_CSV esiste, lo legge dalla prima riga.
    - In caso contrario, usa un fallback con i nomi colonna principali.
    """
    from config_v06_03 import TRACCIATO_REC_DEFAULT_CSV

    try:
        with open(TRACCIATO_REC_DEFAULT_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception:
        return [
            "Informazioni cronologiche",
            "📅 Data dell'evento Near Miss",
            "🕒 Ora indicativa dell'evento Near Miss",
            "📍 Regione",
            "🏫 Luogo/Ambiente dell'evento Near Miss",
            "✍️ Descrizione dell'evento Near Miss",
            "⚠️ Possibili cause",
            "⚠️ Possibili conseguenze",
            "✅ Azione (correttiva) adottata/adottabile o suggerita",
            "🕵️ Segnalato da (facoltativo)",
            "📝 NOTE",
        ]


def _fingerprint(paths: list[str]) -> str:
    """
    Crea un fingerprint veloce di un insieme di file basandosi su:
    - percorso
    - dimensione
    - data ultima modifica

    Utile per capire se l’insieme dei file è cambiato senza ricalcolare hash
    su tutto il contenuto.
    """
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            st_ = os.stat(p)
            h.update(p.encode("utf-8"))
            h.update(str(st_.st_size).encode("utf-8"))
            h.update(str(int(st_.st_mtime)).encode("utf-8"))
        except FileNotFoundError:
            continue
    return h.hexdigest()


def _load_old_sources(vectorstore_dir: str) -> list[str]:
    """
    Carica la lista di sorgenti da sources.json, se esiste.
    Restituisce lista vuota in caso di problemi.
    """
    try:
        with open(
            os.path.join(vectorstore_dir, "sources.json"), "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return []


def valida_csv(df: pd.DataFrame, tracciato: list[str]) -> bool:
    """
    True se le prime len(tracciato) colonne del df sono uguali (in ordine)
    alla lista `tracciato`. Utile per riconoscere il tracciato Near Miss.
    """
    try:
        return list(df.columns)[: len(tracciato)] == tracciato
    except Exception:
        return False


def is_csv_generico(df: pd.DataFrame) -> bool:
    """True se il CSV NON è il tracciato predefinito."""
    return not valida_csv(df, get_tracciato_record_headers())


def is_csv_predef(df: pd.DataFrame) -> bool:
    """
    True se il df corrisponde al tracciato Near Miss predefinito.

    In questa versione: richiede che tutte le colonne definite in
    PREDEF_X_CHOICES siano presenti.
    """
    from config_v06_03 import PREDEF_X_CHOICES

    required = set(PREDEF_X_CHOICES.values())
    cols = set(df.columns)
    return required.issubset(cols)

def sync_csv_dashboard(df: pd.DataFrame, fonte_dati: str):
    """
    Sincronizza il CSV mostrato in dashboard con il RAG quando cambia la fonte
    dei dati (Google Sheet ↔ CSV locale).

    Versione ottimizzata per Colab/Drive:
    - salva sempre dashboard.csv quando la fonte cambia (o al primo avvio)
    - NON ricostruisce l'indice in questa fase (evita blocchi lunghi in RUNNING...)
    - marca l'indice come "da aggiornare" e lo aggiorna solo quando serve
      (invio domanda chat o azione manuale nel pannello RAG).
    """
    # Protezione NameError locale
    _log = logger if 'logger' in globals() else st
    last = st.session_state.get("fonte_corrente", "")
    from config_v06_03 import RAG_CSV_DIR, RAG_CSV_FILE

    sync_necessaria = (fonte_dati != last)
    
    if not sync_necessaria and not st.session_state.get("rag_sync_iniziale_fatto", False):
        rag_csv_path = os.path.join(RAG_CSV_DIR, RAG_CSV_FILE)
        if not os.path.exists(rag_csv_path):
            sync_necessaria = True
        st.session_state["rag_sync_iniziale_fatto"] = True

    if not sync_necessaria:
        return
        
    from Utils_RAG_NearMiss_v06_03 import sincronizza_csv_con_rag

    # Hash del contenuto per capire se i dati sono davvero cambiati
    try:
        csv_payload = df.to_csv(index=False).encode("utf-8")
        new_hash = hashlib.md5(csv_payload).hexdigest()
    except Exception:
        new_hash = ""

    old_hash = st.session_state.get("last_dashboard_hash", "")

    out = sincronizza_csv_con_rag(df)
    if not out:
        st.error("❌ Errore durante il salvataggio del CSV per il RAG.")
        return

    st.session_state["fonte_corrente"] = fonte_dati

    # Primo avvio: memorizzo hash ma non blocco la UI con rebuild immediata.
    if not old_hash:
        st.session_state["last_dashboard_hash"] = new_hash
        st.session_state["rag_needs_rebuild"] = False
        _log.info(f"Sync CSV completata (bootstrap): {fonte_dati}")
        return

    if new_hash and new_hash != old_hash:
        st.session_state["rag_needs_rebuild"] = True
        st.session_state["last_dashboard_hash"] = new_hash
        st.toast("🧠 RAG da aggiornare: lo farò al prossimo invio chat.", icon="ℹ️")
    else:
        st.session_state["rag_needs_rebuild"] = False
        st.session_state["last_dashboard_hash"] = new_hash

    _log.info(f"Sync CSV completata: {fonte_dati}")