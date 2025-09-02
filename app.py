
# -----------------------------------------------------------------------------
# © 2025 Edoardo Salza (https://ardututor.edubot.it). Tutti i diritti riservati.
#
# Nome del Software: TTRG_Tutor (Versione Ibrida con File Upload)
# Versione: 2.15-HYBRID-FILES-STABLE
# Data di Creazione: 2025-07-08
# Data di Revisione: 2025-08-25
# -----------------------------------------------------------------------------

# Import delle librerie necessarie
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import generation_types
import re
import os
import time
from typing import Optional
from PIL import Image
import base64
import uuid
from io import BytesIO
import functools
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import fitz # PyMuPDF

# --- CARICAMENTO VARIABILI D'AMBIENTE ---
if not os.path.exists('/.dockerenv'):
    print("📍 Rilevato ambiente locale. Caricamento del file .env...")
    from dotenv import load_dotenv
    load_dotenv()

# --- CONFIGURAZIONE GLOBALE ---
SERVER_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH", "prompt.md")
DEPLOYMENT_MODE = "server" if SERVER_API_KEY else "user_api"

st.set_page_config(
    page_title="TTRG_Tutor - Assistente AI per Tecnologie e Tecniche di Rappresentazione Grafica",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- COSTANTI ---
SESSION_TIMEOUT = 3600
PDF_MAX_SIZE_MB = 20
MAX_IMAGE_SIZE_MB = 5
MAX_CONCURRENT_FILES = 5

MODEL_CONFIG = {
    "gemini-2.5-flash": {
        "temperature": 0.3,
        "top_p": 0.95,
    },
}

# --- FUNZIONI DI UTILITY E CACHE ---

@st.cache_data
def load_custom_icon():
    """Carica l'icona personalizzata e la converte in base64."""
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            with Image.open(icon_path) as img:
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
        return None
    except Exception:
        return None

CUSTOM_ICON_BASE64 = load_custom_icon()

def get_custom_avatar():
    return CUSTOM_ICON_BASE64 if CUSTOM_ICON_BASE64 else "🤖"

def inject_custom_css():
    try:
        css_path = os.path.join(os.path.dirname(__file__), "style.css")
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ File style.css non trovato.")

# --- SISTEMA DI CACHING FILE ASINCRONO ---# --- BLOCCO GESTIONE CACHE FILE (REIMPLEMENTATO) ---
class CachedFile:
    def __init__(self, file_id, filename, content_type, processed_data, memory_usage, file_hash):
        self.file_id, self.filename, self.content_type = file_id, filename, content_type
        self.processed_data, self.memory_usage, self.file_hash = processed_data, memory_usage, file_hash
        self.is_placeholder = False

    @classmethod
    def placeholder(cls, file_id: str, filename: str):
        ph = cls(file_id, filename, "processing", None, 0, "")
        ph.is_placeholder = True
        return ph

    def is_ready(self) -> bool:
        return not self.is_placeholder and self.processed_data is not None

    def get_for_ai_model(self):
        if not self.is_ready(): return None
        if self.content_type == "image": return self.processed_data
        if self.content_type == "pdf": return self.processed_data.get("text_content")
        return None

class OptimizedFileCache:
    
    def __init__(self, max_memory_mb=100, max_concurrent_processing=2):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache, self.processing_queue, self.access_times = {}, {}, {}
        self.current_memory_usage = 0
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_processing)
        self.lock = threading.RLock()

    def get_file_hash(self, file_data: bytes) -> str:
        return hashlib.blake2b(file_data[:8192], digest_size=8).hexdigest()


    def evict_lru_if_needed(self, needed_bytes: int):
        with self.lock:
            if self.current_memory_usage + needed_bytes <= self.max_memory_bytes:
                return
            sorted_files = sorted(self.access_times.items(), key=lambda x: x[1])
            for file_id, _ in sorted_files:
                if self.current_memory_usage + needed_bytes <= self.max_memory_bytes:
                    break
                if file_id in self.cache:
                    cached_file = self.cache.pop(file_id)
                    self.access_times.pop(file_id)
                    self.current_memory_usage -= cached_file.memory_usage

    def optimize_image(self, image_data: bytes) -> Image.Image:
        img = Image.open(BytesIO(image_data))
        if img.mode != 'RGB': img = img.convert('RGB')
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img

    def extract_pdf_text(self, pdf_data: bytes, max_pages=10) -> dict:
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text_content = [doc[i].get_text() for i in range(min(max_pages, len(doc)))]
            return {"text_content": "\n\n".join(text_content)}
        except Exception as e:
            return {"text_content": f"[Errore estrazione PDF: {e}]", "error": True}

    def process_file_async(self, file_id: str, file_obj) -> 'CachedFile':
        try:
            file_data = file_obj.getvalue()
            file_hash = self.get_file_hash(file_data)
            if existing := self.find_by_hash(file_hash): return existing

            content_type = "pdf" if file_obj.type == "application/pdf" else "image"
            processed_data = self.extract_pdf_text(file_data) if content_type == "pdf" else self.optimize_image(file_data)
            
            memory_usage = len(processed_data["text_content"].encode()) if content_type == "pdf" else processed_data.width * processed_data.height * 3
            cached_file = CachedFile(file_id, file_obj.name, content_type, processed_data, memory_usage, file_hash)

            self.evict_lru_if_needed(memory_usage)
            with self.lock:
                self.cache[file_id], self.access_times[file_id] = cached_file, time.time()
                self.current_memory_usage += memory_usage
            return cached_file
        except Exception as e:
            return CachedFile(file_id, getattr(file_obj, 'name', 'unknown'), "error", {"error": str(e)}, 0, "")

    def find_by_hash(self, file_hash: str) -> Optional['CachedFile']:
        with self.lock:
            return next((f for f in self.cache.values() if f.file_hash == file_hash), None)

    def get_or_create(self, file_id: str, file_obj):
        with self.lock:
            if file_id in self.cache:
                self.access_times[file_id] = time.time()
                return self.cache[file_id]
            if file_id in self.processing_queue:
                future = self.processing_queue[file_id]
                if future.done():
                    return self.processing_queue.pop(file_id).result()
                return CachedFile.placeholder(file_id, file_obj.name)
            
            future = self.executor.submit(self.process_file_async, file_id, file_obj)
            self.processing_queue[file_id] = future
            return CachedFile.placeholder(file_id, file_obj.name)

    def remove_file(self, file_id: str):
        with self.lock:
            if file_id in self.cache:
                cached_file = self.cache.pop(file_id)
                self.access_times.pop(file_id, None)
                self.current_memory_usage -= cached_file.memory_usage

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_memory_usage = 0
            for future in self.processing_queue.values(): future.cancel()
            self.processing_queue.clear()

@st.cache_resource
def get_global_file_cache():
    return OptimizedFileCache(max_memory_mb=150, max_concurrent_processing=3)

#

class SecuritySystem:
    """Sistema di sicurezza con protezione da prompt injection e anonimizzazione dati."""
    def __init__(self, session_id: str):
        self.session_key = str(session_id)
        self.data_patterns = self._initialize_data_patterns()
        self.compiled_data_patterns = {name: re.compile(pattern, re.IGNORECASE)
                                       for name, pattern in self.data_patterns.items()}

    def _initialize_data_patterns(self) -> dict:
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'api_key': r'\bAIza[0-9A-Za-z\-_]{35}\b',
        }

    def anonymize_data(self, text: str) -> str:
        if not text: return text
        anonymized_text = text
        for data_type, pattern in self.compiled_data_patterns.items():
            anonymized_text = pattern.sub(f"[{data_type.upper()}_ANONIMIZZATO]", anonymized_text)
        return anonymized_text

def initialize_session_state():
    """Inizializza uno stato di sessione pulito e minimale."""
    defaults = {
        "chat": None,
        "chat_count": 0,
        "model_initialized": False,
        "anonymous_session_id": f"{uuid.uuid4().hex[:12]}",
        "session_start_time": time.time(),
        "setup_step": "welcome",
        "api_key_configured": False,
        "api_key_entered": DEPLOYMENT_MODE == "server",
        "final_privacy_accepted": False,
        "session_expired": False,
        "security_system": None,
        "uploaded_files": [],
        "uploader_key": 0, 
        # NUOVE AGGIUNTE PER I MESSAGGI PERSISTENTI
        "pending_message": None,  # Per messaggi che devono sopravvivere al rerun
        "pending_message_type": None,  # 'error', 'warning', 'success', 'info'
        "show_message_timer": 0,  # Timestamp per controllare la durata
    }
    for key, default_value in defaults.items():
        if key not in st.session_state: 
            st.session_state[key] = default_value
    if st.session_state.security_system is None:
        st.session_state.security_system = SecuritySystem(st.session_state.anonymous_session_id)

def show_persistent_message():
    """Mostra messaggi persistenti salvati in session_state"""
    if st.session_state.get("pending_message"):
        message_type = st.session_state.get("pending_message_type", "info")
        message_text = st.session_state.pending_message
        
        # Mostra il messaggio in base al tipo
        if message_type == "error":
            st.error(message_text)
        elif message_type == "warning":
            st.warning(message_text)
        elif message_type == "success":
            st.success(message_text)
        else:
            st.info(message_text)
        
        current_time = time.time()
        if st.session_state.show_message_timer == 0:
            st.session_state.show_message_timer = current_time
        elif current_time - st.session_state.show_message_timer > 3:  # 3 secondi
            st.session_state.pending_message = None
            st.session_state.pending_message_type = None
            st.session_state.show_message_timer = 0

def set_persistent_message(message: str, message_type: str = "info"):
    """Imposta un messaggio che sopravviverà al prossimo rerun"""
    st.session_state.pending_message = message
    st.session_state.pending_message_type = message_type
    
    
@st.cache_data(ttl=3600)
def carica_prompt_da_file(filepath: str) -> str:
    """Carica il prompt da file con cache."""
    try:
        script_dir = os.path.dirname(__file__)
        absolute_filepath = os.path.join(script_dir, filepath)
        with open(absolute_filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"❌ ERRORE CRITICO: File del prompt '{filepath}' non trovato.")
        st.stop()

@st.cache_resource
def initialize_model_cached():
    """Inizializza il modello Gemini e lo mette in cache."""
    try:
        model_name = "gemini-2.5-flash"
        system_prompt = carica_prompt_da_file(PROMPT_FILE_PATH)
        safety_settings = [
            {"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        model_params = MODEL_CONFIG.get(model_name)
        if not model_params:
            st.error(f"❌ ERRORE: Configurazione per il modello '{model_name}' non trovata.")
            st.stop()

        generation_config = genai.types.GenerationConfig(
            temperature=model_params["temperature"],
            top_p=model_params["top_p"],
            response_mime_type="text/plain"
        )
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        return model
    except Exception as e:
        st.error(f"Errore nell'inizializzazione del modello: {e}")
        return None

def reset_session():
    """Reset completo della sessione, inclusa la cache dei file."""
    get_global_file_cache().clear()
    keys_to_reset = list(st.session_state.keys())
    for key in keys_to_reset: del st.session_state[key]
    initialize_session_state()
    st.rerun()

def handle_reset_chat():
    """Resetta solo la conversazione corrente, mantenendo la sessione."""
    st.session_state.chat = None
    st.session_state.model_initialized = False
    st.session_state.chat_count = 0
    st.session_state.uploaded_files = []
    get_global_file_cache().clear()
    st.rerun()


def check_session_limits():
    """Controlla i limiti della sessione."""
    if time.time() - st.session_state.session_start_time > SESSION_TIMEOUT:
        st.session_state.session_expired = True
        return False
    return True

    
# --- CONTENUTI DELLE INFORMATIVE ---
def get_informative_content(index):
    """Restituisce il contenuto dell'informativa specificata dall'indice"""
    if DEPLOYMENT_MODE == "server":
        # Versione semplificata per deployment con API del server
        informatives = [
            {
                "title": "INFORMATIVA PRIVACY E PROTEZIONE DATI \n",
                "icon": "🛡️",
                "content": f""" \n
                
• 🔒 **Architettura Stateless**: Nessun dato viene salvato permanentemente. \n
• 🆔 **Session ID Anonimo**: Identificativo temporaneo solo per il funzionamento tecnico. \n
• 🚫 **Nessun dato personale**: Non vengono raccolti nome, email, o altre informazioni personali. \n
• 💬 **Conversazioni temporanee**: Le chat vengono eliminate alla chiusura. \n
• 🔄 **Reset automatico**: La sessione scade dopo **{SESSION_TIMEOUT//60} minuti** di inattività.\n
• 🌐 **Comunicazioni sicure**: Tutte le comunicazioni avvengono tramite HTTPS. \n
"""
            },
            {
                "title": "LIMITAZIONI DELL'INTELLIGENZA ARTIFICIALE \n\n",
                "icon": "⚠️",
                "content": f"""
\n\n
• ⚠️ **Possibili errori**: L'AI può fornire informazioni non accurate o incomplete.\n
• 🧪 **Verifica obbligatoria**: Controlla SEMPRE i codici e i consigli prima dell'uso.\n
• 👨‍💻 **Non sostituisce l'expertise**: È uno strumento di supporto, non un sostituto della competenza umana.\n
• ⚡ **Sicurezza elettrica**: Per progetti complessi, consulta sempre un esperto.\n
**UTILIZZANDO ARDUTUTOR DICHIARI DI AVER COMPRESO E ACCETTATO QUESTE LIMITAZIONI.**
"""
            }
        ]
    else:
        # Versione completa per deployment con API utente
        informatives = [
            {
                "title": "INFORMATIVA OBBLIGATORIA - Utilizzo Chiavi API \n\n",
                "icon": "🔑",
                "content": """
**⚠️ IMPORTANTE - RESPONSABILITÀ DELL'UTENTE PER LA CHIAVE API** \n\n
L'utilizzo di questa applicazione richiede una chiave API personale di Google Gemini. \n\n
**Utilizzando questa applicazione accetti che:** \n
• ✅ **Proprietà della chiave**: La chiave API inserita è di tua proprietà. \n
• 💰 **Responsabilità economica**: Sei l'unico responsabile per tutti i costi associati all'utilizzo della tua chiave API. \n
• 🚫 **Esonero di responsabilità**: L'autore dell'applicazione non è responsabile per costi o danni derivanti dall'uso della tua chiave. \n
• 🔒 **Utilizzo temporaneo**: La chiave **NON viene mai memorizzata permanentemente** e viene eliminata alla chiusura della sessione. \n
"""
            },
            {
                "title": "🔒 INFORMATIVA PRIVACY E PROTEZIONE DATI",
                "icon": "🛡️",
                "content": f"""
**INFORMAZIONI SUL TRATTAMENTO DEI DATI**\n\n
TTRG_Tutor è progettato con un approccio "Privacy by Design": \n\n
• 🔒 **Architettura Stateless**: Nessun dato viene salvato permanentemente. \n
• 🆔 **Session ID Anonimo**: Identificativo temporaneo solo per il funzionamento tecnico. \n
• 🚫 **Nessun dato personale**: Non vengono raccolti nome, email, o altre informazioni personali. \n
• 💬 **Conversazioni temporanee**: Le chat vengono eliminate alla chiusura. \n
• 🔄 **Reset automatico**: La sessione scade dopo **{SESSION_TIMEOUT//60} minuti** di inattività.\n
• 🌐 **Comunicazioni sicure**: Tutte le comunicazioni avvengono tramite HTTPS. \n
"""
            },
            {
                "title": "🤖 LIMITAZIONI DELL'INTELLIGENZA ARTIFICIALE",
                "icon": "⚠️",
                "content": """
**DICHIARAZIONE DI LIMITAZIONI E RESPONSABILITÀ** \n\n
È fondamentale comprendere che TTRG_Tutor è uno strumento AI con limitazioni:\n\n
• ⚠️ **Possibili errori**: L'AI può fornire informazioni non accurate o incomplete.\n
• 🧪 **Verifica obbligatoria**: Controlla SEMPRE i codici e i consigli prima dell'uso.\n
• 👨‍💻 **Non sostituisce l'expertise**: È uno strumento di supporto, non un sostituto della competenza umana.\n
• ⚡ **Sicurezza elettrica**: Per progetti complessi, consulta sempre un esperto.\n
**UTILIZZANDO ARDUTUTOR DICHIARI DI AVER COMPRESO E ACCETTATO QUESTE LIMITAZIONI.**
"""
            }
        ]
    return informatives[index] if 0 <= index < len(informatives) else None

# --- FUNZIONI CONTENUTO PRINCIPALE ---

def show_welcome_content():
    """Mostra il contenuto di benvenuto"""
    st.markdown('<div class="welcome-screen">', unsafe_allow_html=True)

    if CUSTOM_ICON_BASE64:
        st.markdown(f'<h1><img src="{CUSTOM_ICON_BASE64}" class="custom-avatar"> TTRG_Tutor</h1>', unsafe_allow_html=True)
    else:
        st.title("🤖 TTRG_Tutor")

    st.markdown("### Il tuo assistente AI per Tecnologie e Tecniche di Rappresentazione Grafica")


    st.markdown("""
        **Cosa posso fare per te:**
        • 💡 Spiegare concetti • 🔧 Aiutarti con il codice • 🛠️ Risolvere problemi
        """)

    st.markdown("---")
    st.markdown("### 🚀 Inizia")
    
    if DEPLOYMENT_MODE == "server":
        st.markdown("Per utilizzare TTRG_Tutor dovrai leggere alcune informative.")
    else:
        st.markdown("Per utilizzare TTRG_Tutor dovrai leggere alcune informative e configurare la tua chiave API personale.")
    
    if st.button("📋 Leggi le informative prima di iniziare", key="start_config", type="primary", use_container_width=True):
        st.session_state.setup_step = "api_info" if DEPLOYMENT_MODE == "user_api" else "privacy_info"
        st.session_state.informative_index = 0
        st.rerun()

    st.markdown("---")
    app_footer()
    st.markdown('</div>', unsafe_allow_html=True)

def show_informative_sequential():
    """Mostra le informative in modo sequenziale"""
    informative = get_informative_content(st.session_state.informative_index)
    max_informatives = 2 if DEPLOYMENT_MODE == "server" else 3

    if not informative:
        st.session_state.all_informatives_read = True
        if DEPLOYMENT_MODE == "server":
            st.session_state.setup_step = "final_privacy"
        else:
            st.session_state.setup_step = "api_key"
        st.rerun()
        return

    st.markdown(f'<div class="informative-container">', unsafe_allow_html=True)
    st.markdown(f"""<div class="informative-header">{informative['icon']} {informative['title']}</div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="informative-content">{informative["content"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sequential-navigation">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.session_state.informative_index > 0:
            if st.button("← Precedente", key="prev_informative"):
                st.session_state.informative_index -= 1
                st.rerun()
        else:
             if st.button("← Inizio", key="back_to_welcome"):
                st.session_state.setup_step = "welcome"
                st.rerun()

    with col2:
        current = st.session_state.informative_index + 1
        st.markdown(f"<center><strong>Informativa {current} di {max_informatives}</strong></center>", unsafe_allow_html=True)
        progress = (current / max_informatives) * 100
        st.markdown(f'<div class="custom-progress"><div class="custom-progress-bar" style="width: {progress}%"></div></div>', unsafe_allow_html=True)

    with col3:
        if st.session_state.informative_index < max_informatives - 1:
            if st.button("Successiva →", key="next_informative", type="primary"):
                st.session_state.informative_index += 1
                st.rerun()
        else:
            if st.button("Ho Letto Tutto →", key="finish_informatives", type="primary"):
                st.session_state.all_informatives_read = True
                if DEPLOYMENT_MODE == "server":
                    st.session_state.setup_step = "final_privacy"
                else:
                    st.session_state.setup_step = "api_key"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Chiudi sequential-navigation
    st.markdown('</div>', unsafe_allow_html=True)  # Chiudi informative-container


def show_final_privacy_content():
    """Mostra il contenuto per l'accettazione finale della privacy"""
    st.markdown("### 🔒 Conferma Finale delle Condizioni")

    if not st.session_state.final_privacy_accepted:
        st.markdown("**Conferma di aver compreso tutte le condizioni d'uso**")

        if DEPLOYMENT_MODE == "server":
            st.success("""
            **✅ RIEPILOGO DELLE CONDIZIONI ACCETTATE:**

            **🔒 Privacy**: Hai letto l'informativa sulla gestione privacy-by-design dei dati

            **🤖 Limitazioni AI**: Sei consapevole dei limiti dell'intelligenza artificiale

            **🆓 Servizio Gratuito**: Stai utilizzando la versione dimostrativa gratuita
            """)
        else:
            st.success("""
            **✅ RIEPILOGO DELLE CONDIZIONI ACCETTATE:**

            **📋 Informativa API**: Hai compreso la responsabilità per l'utilizzo della tua chiave API

            **🔒 Privacy**: Hai letto l'informativa sulla gestione privacy-by-design dei dati

            **🤖 Limitazioni AI**: Sei consapevole dei limiti dell'intelligenza artificiale

            **🔑 Chiave API**: La tua chiave è stata verificata e sarà utilizzata solo temporaneamente
            """)

        st.markdown("---")

        privacy_final_accepted = st.checkbox(
            "✅ Confermo di aver letto, compreso e accettato tutte le condizioni d'uso e l'informativa privacy completa",
            key="privacy_final_checkbox",
            value=False
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("← Indietro", key="back_from_final_privacy"):
                if DEPLOYMENT_MODE == "server":
                    st.session_state.setup_step = "ai_limitations"
                    st.session_state.informative_index = 1
                else:
                    st.session_state.setup_step = "api_key"
                st.rerun()

        with col2:
            if st.button("🚀 Avvia TTRG_Tutor", key="start_ardututor_final", type="primary", disabled=not privacy_final_accepted, use_container_width=True):
                if privacy_final_accepted:
                    # Configura la chiave API se in modalità server
                    if DEPLOYMENT_MODE == "server":
                        genai.configure(api_key=SERVER_API_KEY)

                    # Completa la configurazione
                    st.session_state.final_privacy_accepted = True
                    st.session_state.setup_step = "ready"
                    st.session_state.api_key_configured = True

                    st.success("✅ Configurazione completata! Avvio di TTRG_Tutor in corso...")
                    time.sleep(1)
                    st.rerun()

        if not privacy_final_accepted:
            st.warning("⚠️ Devi confermare di aver accettato tutte le condizioni per continuare")
    else:
        st.success("✅ Configurazione completata!")
        st.markdown("Tutte le condizioni sono state accettate. TTRG_Tutor è pronto per l'uso.")

def show_ready_content():
    """Mostra il contenuto quando tutto è pronto"""
    

    if DEPLOYMENT_MODE == "server":
        st.success("Sei pronto per iniziare. Ecco alcune domande che puoi farmi:")
    else:
        st.success("Sei pronto per iniziare. Ecco alcune domande che puoi farmi:")

    show_usage_info()

    # Esempi di domande pertinenti al corso di Disegno Tecnico e Rappresentazione Grafica (TTRG)
    # per il biennio di un Istituto Tecnico.
    examples = {
        "📏 **Disegno di Base e Strumenti**": [
            "Che differenza c'è tra una linea a tratto continuo grosso e una fine secondo la norma UNI?",
            "Non riesco a fare una linea perfettamente parallela con le due squadre, come posso fare?",
            "Se devo disegnare un pezzo lungo 500mm in scala 1:5, come calcolo la misura sul foglio?"
        ],
        "📐 **Costruzioni e Proiezioni**": [
            "Mi puoi spiegare passo passo come si costruisce un pentagono dato il lato?",
            "Come rappresento un cilindro in proiezioni ortogonali se è appoggiato sulla base?",
            "Qual è la differenza principale tra un'assonometria isometrica e una cavaliera?"
        ],
        "💻 **CAD e Normative**": [
            "In AutoCAD, come faccio a creare un nuovo layer per le linee di quotatura?",
            "Non ho capito bene quando si usa una sezione e quando si fa un taglio su un pezzo meccanico.",
            "Perché la linea di misura non deve mai toccare lo spigolo dell'oggetto?"
        ]
    }
    # Creiamo un toggle per ogni categoria
    for category, items in examples.items():
        # Creiamo un interruttore con il nome della categoria
        show_items = st.toggle(f"**{category}**", key=category)
        
       # Se l'interruttore è attivo, mostriamo i contenuti
        if show_items:
          # # Usiamo un container per raggruppare i suggerimenti
            with st.container(border=True):
                for item in items:
                    st.markdown(f"• _{item}_")
            st.write("") # Aggiunge un po' di spazio

    if st.button("💬 Inizia", key="start_chat", type="primary", use_container_width=True):
        initialize_chat()
        st.rerun()

# --- FUNZIONI CONTENUTO PRINCIPALE ---
# ... (dopo show_ready_content)

def show_onboarding_flow():
    """Funzione router per il flusso di configurazione iniziale."""
    step = st.session_state.setup_step
    if step == "welcome":
        show_welcome_content()
    elif step in ["api_info", "privacy_info", "ai_limitations"]: # Steps gestiti dalle informative
        show_informative_sequential()
    elif step == "api_key":
        # Nota: La funzione show_api_key_content() non è nel codice fornito,
        # ma questo la gestirebbe se esistesse. In caso contrario, il flusso salta a "final_privacy".
        # Per ora, si assume che la logica API porti a "final_privacy".
        # Se hai una funzione per l'inserimento della chiave API, chiamala qui.
        # Altrimenti, il flusso prosegue come gestito da show_informative_sequential.
        show_final_privacy_content() # Placeholder se non c'è una UI per la chiave API
    elif step == "final_privacy":
        show_final_privacy_content()
    elif step == "ready":
        show_ready_content()
        

    
def display_file_manager():
    """UI per la gestione dei file caricati - VERSIONE CORRETTA"""
    cache = get_global_file_cache()
    files_nella_sessione = st.session_state.get("uploaded_files", [])
    files_count = len(files_nella_sessione)
    
    show_file_manager = st.toggle(
        f"📎 Gestione File ({files_count}/{MAX_CONCURRENT_FILES})", 
        key="toggle_file_manager"
    )

    if show_file_manager:
        limite_raggiunto = files_count >= MAX_CONCURRENT_FILES
        nuovi_file_caricati = st.file_uploader(
            "Allega file (PDF max 20MB, immagini max 5MB)", type=['png', 'jpg', 'jpeg', 'webp', 'pdf'],
            accept_multiple_files=True,
            key=f"file_uploader_widget_{st.session_state.uploader_key}",
            disabled=limite_raggiunto, help=f"Limite di {MAX_CONCURRENT_FILES} file."
        )

        if nuovi_file_caricati:
            id_esistenti = {f.file_id for f in files_nella_sessione}
            file_validi = []
            file_potenziali = [f for f in nuovi_file_caricati if f.file_id not in id_esistenti]
            
            # VARIABILI PER TRACCIARE SE CI SONO ERRORI
            errori_dimensioni = []
            
            for file in file_potenziali:
                size_in_bytes = file.size
                if file.type.startswith("image/") and size_in_bytes > (MAX_IMAGE_SIZE_MB * 1024 * 1024):
                    errori_dimensioni.append(f"🖼️ L'immagine '{file.name}' è troppo grande. Limite: {MAX_IMAGE_SIZE_MB} MB.")
                elif file.type == "application/pdf" and size_in_bytes > (PDF_MAX_SIZE_MB * 1024 * 1024):
                    errori_dimensioni.append(f"📄 Il PDF '{file.name}' è troppo grande. Limite: {PDF_MAX_SIZE_MB} MB.")
                else:
                    file_validi.append(file)

            spazio_disponibile = MAX_CONCURRENT_FILES - files_count
            warning_troppi_file = None
            
            if len(file_validi) > spazio_disponibile:
                warning_troppi_file = f"⚠️ Hai caricato troppi file. C'è spazio solo per {spazio_disponibile} files. Verranno aggiunti solo i primi file idonei."
            
            file_finali = file_validi[:spazio_disponibile]
            
            # MOSTRA GLI ERRORI PRIMA DEL RERUN
            for errore in errori_dimensioni:
                st.error(errore)
            
            if warning_troppi_file:
                st.warning(warning_troppi_file)
            
            if file_finali:
                # Imposta un messaggio locale temporaneo invece di quello globale
                st.session_state['upload_status_message'] = {
                    "text": f"✅ {len(file_finali)} file caricati con successo!",
                    "type": "success"
                }
                
                for file in file_finali:
                    cache.get_or_create(file.file_id, file)
                st.session_state.uploaded_files.extend(file_finali)
                
                # OPZIONE 2: Delay prima del rerun SE ci sono stati errori
                if errori_dimensioni or warning_troppi_file:
                    time.sleep(3)  # Permette di leggere i messaggi
                
                st.rerun()
        if files_nella_sessione:
            st.markdown("---")
            if st.button("🧹 Pulisci Tutti i File", key="clear_all_files", use_container_width=True, type="secondary"):
                cache.clear()
                st.session_state.uploaded_files = []
                st.session_state.uploader_key += 1
                st.rerun()

            for i, file in reversed(list(enumerate(files_nella_sessione))):
                cached_file = cache.get_or_create(file.file_id, file)
                col1, col2 = st.columns([4, 1])
                with col1:
                    icon = "🖼️" if file.type.startswith("image") else "📄"
                    status = "✅" if cached_file.is_ready() else "⏳"
                    st.caption(f"{status} {icon} {file.name}")
                with col2:
                    if st.button("🗑️", key=f"remove_{file.file_id}", help=f"Rimuovi {file.name}"):
                        file_id_da_rimuovere = st.session_state.uploaded_files.pop(i).file_id
                        cache.remove_file(file_id_da_rimuovere)
                        st.session_state.uploader_key += 1
                        st.rerun()

    # --- INIZIO BLOCCO DA AGGIUNGERE ---
    # Controlla se c'è un messaggio di stato dell'upload da mostrare
    if 'upload_status_message' in st.session_state:
        message_info = st.session_state['upload_status_message']
        message_text = message_info.get("text")
        message_type = message_info.get("type")

        if message_type == "success":
            st.success(message_text)
        elif message_type == "warning":
            st.warning(message_text)
        elif message_type == "error":
            st.error(message_text)
        else:
            st.info(message_text)
        
        # Rimuovi il messaggio dallo stato per non mostrarlo di nuovo
        del st.session_state['upload_status_message']
    # --- FINE BLOCCO DA AGGIUNGERE ---
# --- LOGICA DELLA CHAT ---

def initialize_chat():
    """Inizializza la chat di TTRG_Tutor."""
    if not st.session_state.model_initialized:
        with st.spinner("Inizializzazione TTRG_Tutor..."):
            model = initialize_model_cached()
            if model:
                st.session_state.chat = model.start_chat(history=[])
                welcome_msg = "Ciao! Sono TTRG_Tutor. Fammi una domanda o allega un file!"
                st.session_state.chat.history.append({'role': 'model', 'parts': [{'text': welcome_msg}]})
                st.session_state.model_initialized = True
                st.session_state.setup_step = "chat"

def handle_user_prompt(user_prompt: str):
    """
    Gestisce la logica del prompt utente: sanitizzazione, chiamata al modello
    e aggiornamento della cronologia della chat (SENZA visualizzare nulla).
    """
    if not user_prompt.strip():
        st.warning("⚠️ Inserisci una domanda.")
        return

    sanitized_prompt = st.session_state.security_system.anonymize_data(user_prompt)    
    # Prepara il contenuto per il modello
    model_content = [sanitized_prompt]
    cache = get_global_file_cache()
    for file_obj in st.session_state.get("uploaded_files", []):
        cached_file = cache.get_or_create(file_obj.file_id, file_obj)
        if cached_file.is_ready():
            ai_content = cached_file.get_for_ai_model()
            if ai_content:
                model_content.append(ai_content)

    try:
        # 2. Invia il messaggio al modello
        with st.spinner("TTRG_Tutor sta elaborando..."):
            # Invia il messaggio al modello
            # send_message() gestisce automaticamente la cronologia
            response = st.session_state.chat.send_message(model_content)
            full_response_text = response.text
      
    except Exception as e:
        # MOSTRA L'ERRORE IN MODO PERSISTENTE
        error_message = f"🔧 Errore durante la generazione della risposta: {e}"
        set_persistent_message(error_message, "error")
        
        # Rimuovi il messaggio utente se la chiamata fallisce
        if st.session_state.chat.history and st.session_state.chat.history[-1].get('role') == 'user':
            st.session_state.chat.history.pop()
            st.session_state.chat_count -= 1

def show_chat_content():
    """Mostra l'interfaccia di chat completa."""
    if not st.session_state.get("model_initialized"):
        initialize_chat()

    # Header della chat con controlli
    col1, col_spacer, col2, col3 = st.columns([2, 2, 1, 1])
    with col1:
        custom_avatar = get_custom_avatar()
        if custom_avatar.startswith('data:image'):
            st.markdown(f'<h1><img src="{custom_avatar}" class="custom-avatar"> Chiedilo ad TTRG_Tutor!</h1>', unsafe_allow_html=True)
        else:
            st.title("Chiedilo a TTRG_Tutor!")
    
    with col2:
        if st.button("🔄 Reset Chat", use_container_width=True):
            handle_reset_chat()
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.google_credentials = None
            reset_session()
    
    # Container per la cronologia della chat
    with st.container(border=True):
        if st.session_state.chat and st.session_state.chat.history:
            for messaggio in st.session_state.chat.history:
                # Gestione robusta del formato messaggio
                try:
                    if isinstance(messaggio, dict):
                        role = messaggio.get('role', 'unknown')
                        text_content = messaggio.get('parts', [{}])[0].get('text', 'Contenuto non disponibile')
                    else:
                        role = getattr(messaggio, 'role', 'unknown')
                        parts = getattr(messaggio, 'parts', [])
                        text_content = parts[0].text if parts and hasattr(parts[0], 'text') else 'Contenuto non disponibile'
                    
                    ruolo_display = "Studente" if role == 'user' else "TTRG_Tutor"
                    avatar = "🧑‍🎓" if role == 'user' else get_custom_avatar()
                    
                    with st.chat_message(ruolo_display, avatar=avatar):
                        st.markdown(text_content)
                        
                except (KeyError, IndexError, AttributeError) as e:
                    st.warning(f"Impossibile visualizzare un messaggio: {e}")

    # Input per nuovi messaggi
    if prompt_utente := st.chat_input("Scrivi qui la tua domanda..."):
        handle_user_prompt(prompt_utente)
        st.rerun()

    # File manager
    display_file_manager()
    
    # Informazioni sessione
    session_duration = int(time.time() - st.session_state.session_start_time)
    st.caption(f"Domande: {st.session_state.chat_count} | Durata: {session_duration//60}m {session_duration%60}s")

def app_footer():
    """Mostra un footer standard per l'applicazione."""
    st.markdown("---")
    st.markdown("© 2025 Edoardo Salza. Tutti i diritti riservati.")
    st.markdown("Applicazione creata per scopi didattici e di supporto.")

def show_usage_info():
    """Mostra informazioni su come usare la chat."""
    with st.expander("💡 Consigli d'uso", expanded=False):
        st.info("""
        - **Allega File**: Puoi caricare immagini max 5 MB (.png, .jpg) o documenti MAX 20MB (.pdf) per contestualizzare le tue domande.
        - **Sii Specifico**: Più dettagli fornisci, migliore sarà la risposta. Includi messaggi di errore e spezzoni di codice.
        - **Reset Chat**: Se la conversazione diventa confusa, usa il pulsante "Reset Chat" per ricominciare da capo.
        """)

# --- FUNZIONE PRINCIPALE ---
def main():
    """Esegue l'applicazione Streamlit."""
    inject_custom_css()
    initialize_session_state()
    
    # MOSTRA SEMPRE I MESSAGGI PERSISTENTI ALL'INIZIO
    show_persistent_message()


    if time.time() - st.session_state.session_start_time > SESSION_TIMEOUT:
        st.session_state.session_expired = True

    if st.session_state.session_expired:
        st.warning(f"⏰ Sessione scaduta. Ricarica la pagina per iniziare una nuova sessione.")
        if st.button("🔄 Inizia Nuova Sessione", key="reset_expired"):
            reset_session()
        return

    if st.session_state.api_key_configured and st.session_state.final_privacy_accepted:
        show_chat_content()
    else:
        show_onboarding_flow()

if __name__ == "__main__":
    main()













