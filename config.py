"""
PAPERWISE AI APEX - Centralized Configuration Module
Provides environment-aware configuration, validation boundaries, and single-source model definition.
All credentials are loaded exclusively server-side from local .env or Streamlit secrets.
No keys, tokens, or credential masks are ever exposed to the frontend.
"""

from dataclasses import dataclass
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load .env strictly server-side from workspace root
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
except ImportError:
    pass


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve secrets server-side prioritizing:
    1. OS environment / .env file
    2. Streamlit secrets (st.secrets) if running within Streamlit
    3. Default value
    """
    val = os.getenv(key)
    if val and val.strip():
        return val.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            sec_val = st.secrets[key]
            if sec_val and str(sec_val).strip():
                return str(sec_val).strip()
    except Exception:
        pass

    return default


def is_api_configured() -> bool:
    """Check if the Gemini API Key is configured on the server without leaking value."""
    key = get_secret("GEMINI_API_KEY")
    return bool(key and len(key.strip()) > 5)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration, brand identity, and document constraints."""
    app_name: str = "PAPERWISE AI APEX"
    edition: str = "APEX v1.0"
    tagline: str = "Research deserves better than a skim."
    subtagline: str = "Transform complex academic knowledge into clarity."

    # Single centralized model configuration
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    model_display_name: str = "Gemini 3.5 Flash-Lite"

    # PDF Processing Boundaries
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "30"))
    max_page_count: int = int(os.getenv("MAX_PAGE_COUNT", "60"))
    min_word_count_threshold: int = int(os.getenv("MIN_WORD_COUNT_THRESHOLD", "50"))
    chunk_page_size: int = int(os.getenv("CHUNK_PAGE_SIZE", "5"))
    synthesis_page_threshold: int = int(os.getenv("SYNTHESIS_PAGE_THRESHOLD", "8"))

    # Visual Design Palette
    bg_primary: str = "#06080F"
    bg_surface: str = "#0D121F"
    bg_card: str = "#111827"
    border_subtle: str = "rgba(255, 255, 255, 0.08)"
    border_accent: str = "rgba(139, 92, 246, 0.35)"
    accent_violet: str = "#8B5CF6"
    accent_cyan: str = "#06B6D4"
    accent_emerald: str = "#10B981"
    text_primary: str = "#F8FAFC"
    text_muted: str = "#94A3B8"
    text_dim: str = "#64748B"


# Global singleton instance
config = AppConfig()
