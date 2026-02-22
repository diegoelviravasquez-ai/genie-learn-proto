"""
CONFIG.PY — Configuraciones del Sistema GENIE Learn
=====================================================
Centraliza TODAS las configuraciones del ecosistema:
  - PedagogicalConfig: reglas pedagógicas del docente
  - SystemConfig: parámetros técnicos del sistema
  - RAGConfig: configuración del pipeline RAG
  - LLMConfig: parámetros de los modelos de lenguaje
  - AnalyticsConfig: configuración de analytics

Re-exporta desde middleware.py para compatibilidad con código existente.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum

from middleware import PedagogicalConfig, InteractionLog

# Re-export principal
__all__ = [
    "PedagogicalConfig", 
    "InteractionLog",
    "SystemConfig",
    "RAGConfig",
    "LLMConfig",
    "AnalyticsConfig",
    "get_config",
    "PRESETS",
]


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM CONFIG — Parámetros técnicos globales
# ═══════════════════════════════════════════════════════════════════════

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SystemConfig:
    """Configuración técnica del sistema."""
    
    # Entorno
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    
    # Base de datos
    database_url: str = "sqlite:///./genie_learn.db"
    database_echo: bool = False
    
    # Redis (para rate limiting en producción)
    redis_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    
    # Seguridad
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "dev-secret-change-in-production"))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # LTI
    lti_enabled: bool = False
    lti_platform_id: Optional[str] = None
    lti_client_id: Optional[str] = None
    lti_deployment_id: Optional[str] = None
    lti_keyset_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Carga configuración desde variables de entorno."""
        return cls(
            environment=Environment(os.getenv("ENVIRONMENT", "development")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_workers=int(os.getenv("API_WORKERS", "4")),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "60")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./genie_learn.db"),
            database_echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            redis_url=os.getenv("REDIS_URL"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            jwt_secret=os.getenv("JWT_SECRET", "dev-secret-change-in-production"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            lti_enabled=os.getenv("LTI_ENABLED", "false").lower() == "true",
            lti_platform_id=os.getenv("LTI_PLATFORM_ID"),
            lti_client_id=os.getenv("LTI_CLIENT_ID"),
            lti_deployment_id=os.getenv("LTI_DEPLOYMENT_ID"),
            lti_keyset_url=os.getenv("LTI_KEYSET_URL"),
        )


# ═══════════════════════════════════════════════════════════════════════
# RAG CONFIG — Configuración del pipeline de retrieval
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RAGConfig:
    """Configuración del pipeline RAG."""
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Retrieval
    top_k: int = 3
    similarity_threshold: float = 0.25
    
    # ChromaDB
    chroma_persist_directory: Optional[str] = None
    chroma_collection_name: str = "course_materials"
    
    # Re-ranking (opcional)
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3


# ═══════════════════════════════════════════════════════════════════════
# LLM CONFIG — Configuración de modelos de lenguaje
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuración de modelos de lenguaje."""
    
    # Provider y modelo
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    
    # Parámetros de generación
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    
    # Timeouts
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Fallback
    fallback_provider: Optional[LLMProvider] = None
    fallback_model: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Detecta el provider disponible según API keys."""
        if os.getenv("OPENAI_API_KEY"):
            return cls(provider=LLMProvider.OPENAI, model_name="gpt-4o-mini")
        elif os.getenv("ANTHROPIC_API_KEY"):
            return cls(provider=LLMProvider.ANTHROPIC, model_name="claude-sonnet-4-20250514")
        else:
            return cls(provider=LLMProvider.MOCK, model_name="mock-demo")


# ═══════════════════════════════════════════════════════════════════════
# ANALYTICS CONFIG — Configuración de analytics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnalyticsConfig:
    """Configuración del sistema de analytics."""
    
    # Detección de copy-paste
    copypaste_threshold: float = 0.5
    copypaste_min_length: int = 50
    
    # Análisis cognitivo
    bloom_confidence_threshold: float = 0.6
    icap_enabled: bool = True
    
    # Neurodivergencia
    nd_detection_enabled: bool = True
    nd_min_interactions: int = 5
    
    # Consolidación
    consolidation_window_hours: int = 72
    consolidation_min_sessions: int = 2
    
    # Export
    export_format: Literal["csv", "json", "parquet"] = "csv"
    anonymize_exports: bool = True
    k_anonymity_level: int = 5


# ═══════════════════════════════════════════════════════════════════════
# PRESETS PEDAGÓGICOS
# ═══════════════════════════════════════════════════════════════════════

PRESET_EXAMEN = PedagogicalConfig(
    max_daily_prompts=3,
    scaffolding_mode="socratic",
    block_direct_solutions=True,
    forced_hallucination_pct=0.0,
    use_rag=True,
    no_context_behavior="refuse",
    role_play="Eres un evaluador estricto. No ayudes a resolver, solo clarifica el enunciado.",
)

PRESET_REPASO_LIBRE = PedagogicalConfig(
    max_daily_prompts=50,
    scaffolding_mode="direct",
    block_direct_solutions=False,
    forced_hallucination_pct=0.0,
    use_rag=True,
    no_context_behavior="general",
    role_play="Eres un tutor paciente que quiere que el estudiante entienda profundamente.",
)

PRESET_LECTURA_CRITICA = PedagogicalConfig(
    max_daily_prompts=20,
    scaffolding_mode="hints",
    block_direct_solutions=True,
    forced_hallucination_pct=0.15,
    use_rag=True,
    no_context_behavior="refuse",
    role_play="Responde con información que puede contener errores sutiles. El estudiante debe verificar.",
)

PRESET_EXPLORATORIO = PedagogicalConfig(
    max_daily_prompts=30,
    scaffolding_mode="hints",
    block_direct_solutions=False,
    forced_hallucination_pct=0.0,
    use_rag=True,
    no_context_behavior="general",
    role_play="Eres un guía curioso que fomenta la exploración y las preguntas abiertas.",
)

PRESETS = {
    "examen": PRESET_EXAMEN,
    "repaso": PRESET_REPASO_LIBRE,
    "lectura_critica": PRESET_LECTURA_CRITICA,
    "exploratorio": PRESET_EXPLORATORIO,
}


def get_preset(name: str) -> PedagogicalConfig:
    """Devuelve un preset de configuración por nombre."""
    if name not in PRESETS:
        raise ValueError(f"Preset '{name}' no existe. Opciones: {list(PRESETS.keys())}")
    return PRESETS[name]


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    """Configuración completa de la aplicación."""
    system: SystemConfig = field(default_factory=SystemConfig)
    pedagogical: PedagogicalConfig = field(default_factory=PedagogicalConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Carga toda la configuración desde el entorno."""
        return cls(
            system=SystemConfig.from_env(),
            pedagogical=PedagogicalConfig(),
            rag=RAGConfig(),
            llm=LLMConfig.from_env(),
            analytics=AnalyticsConfig(),
        )


# Singleton de configuración
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Obtiene la configuración global (singleton)."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Resetea la configuración (para tests)."""
    global _config
    _config = None
