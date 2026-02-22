"""
RAG_ENGINE.PY — Motor de Retrieval-Augmented Generation
========================================================
Wrapper sobre rag_pipeline.py que provee la interfaz RAGEngine
esperada por api.py y otros módulos del ecosistema.

Este módulo unifica las dos implementaciones de RAG:
  - SimpleRAGPipeline: TF-IDF básico (sin API key)
  - OpenAIRAGPipeline: ChromaDB + OpenAI embeddings

La clase RAGEngine es el punto de entrada único para el resto del sistema.
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from rag_pipeline import (
    get_rag_pipeline,
    SimpleRAGPipeline,
    OpenAIRAGPipeline,
    SAMPLE_COURSE_CONTENT,
)

logger = logging.getLogger(__name__)

# Re-export para compatibilidad
__all__ = [
    "RAGEngine",
    "RAGResult",
    "SAMPLE_COURSE_CONTENT",
    "get_rag_engine",
]


@dataclass
class RAGResult:
    """Resultado de una consulta RAG."""
    text: str
    source: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "metadata": self.metadata,
        }


class RAGEngine:
    """
    Motor RAG unificado para GENIE Learn.
    
    Uso:
        engine = RAGEngine()
        engine.ingest_pdf("/path/to/doc.pdf")
        results = engine.retrieve("¿Cómo funciona un bucle for?")
        context = engine.build_context("¿Cómo funciona un bucle for?")
    
    El backend (Simple vs OpenAI) se selecciona automáticamente
    según la disponibilidad de OPENAI_API_KEY.
    """
    
    def __init__(
        self,
        use_openai: Optional[bool] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        persist_directory: Optional[str] = None,
    ):
        """
        Inicializa el motor RAG.
        
        Args:
            use_openai: Forzar uso de OpenAI (None = auto-detectar)
            chunk_size: Tamaño de chunks en caracteres
            chunk_overlap: Overlap entre chunks
            persist_directory: Directorio para persistir ChromaDB
        """
        if use_openai is None:
            use_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        self._pipeline = get_rag_pipeline(use_openai=use_openai)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._persist_directory = persist_directory
        
        logger.info(
            "RAGEngine inicializado: backend=%s, chunk_size=%d, overlap=%d",
            "OpenAI" if use_openai else "Simple",
            chunk_size,
            chunk_overlap,
        )
    
    @property
    def is_loaded(self) -> bool:
        """Indica si hay documentos cargados."""
        return self._pipeline.is_loaded
    
    @property
    def backend_type(self) -> str:
        """Tipo de backend activo."""
        if isinstance(self._pipeline, OpenAIRAGPipeline):
            return "openai"
        return "simple"
    
    def ingest_text(
        self,
        text: str,
        source: str = "documento",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> int:
        """
        Ingesta texto crudo → chunking → indexación.
        
        Returns:
            Número de chunks creados
        """
        chunk_size = chunk_size or self._chunk_size
        overlap = overlap or self._chunk_overlap
        
        n_chunks = self._pipeline.ingest_text(text, source, chunk_size, overlap)
        logger.info("Ingestado '%s': %d chunks", source, n_chunks)
        return n_chunks
    
    def ingest_pdf(
        self,
        pdf_path: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> int:
        """
        Ingesta PDF → extracción de texto → chunking → indexación.
        
        Returns:
            Número de chunks creados
        """
        chunk_size = chunk_size or self._chunk_size
        overlap = overlap or self._chunk_overlap
        
        n_chunks = self._pipeline.ingest_pdf(pdf_path, chunk_size, overlap)
        logger.info("Ingestado PDF '%s': %d chunks", pdf_path, n_chunks)
        return n_chunks
    
    def ingest_directory(
        self,
        directory: str,
        extensions: List[str] = [".pdf", ".txt", ".md"],
    ) -> Dict[str, int]:
        """
        Ingesta todos los archivos de un directorio.
        
        Returns:
            Dict con {filename: n_chunks} por archivo procesado
        """
        import os
        results = {}
        
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue
            
            filepath = os.path.join(directory, filename)
            
            if ext == ".pdf":
                n = self.ingest_pdf(filepath)
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                n = self.ingest_text(content, filename)
            
            results[filename] = n
        
        logger.info("Directorio ingestado: %d archivos, %d chunks totales",
                    len(results), sum(results.values()))
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[RAGResult]:
        """
        Recupera los chunks más relevantes para una query.
        
        Args:
            query: Texto de búsqueda
            top_k: Número máximo de resultados
            threshold: Score mínimo para incluir (0-1)
        
        Returns:
            Lista de RAGResult ordenados por relevancia
        """
        raw_results = self._pipeline.retrieve(query, top_k)
        
        results = []
        for r in raw_results:
            if r.get("score", 0) >= threshold:
                results.append(RAGResult(
                    text=r["text"],
                    source=r.get("source", "unknown"),
                    chunk_index=r.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                ))
        
        logger.debug("Retrieve '%s': %d resultados", query[:50], len(results))
        return results
    
    def build_context(
        self,
        query: str,
        top_k: int = 3,
        max_length: int = 3000,
        include_sources: bool = True,
    ) -> str:
        """
        Construye el contexto RAG para inyectar en el prompt del LLM.
        
        Args:
            query: Texto de búsqueda
            top_k: Número de chunks a incluir
            max_length: Longitud máxima del contexto
            include_sources: Incluir referencias a fuentes
        
        Returns:
            Contexto formateado como string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, r in enumerate(results, 1):
            if include_sources:
                part = f"[Fragmento {i} — {r.source}, sección {r.chunk_index}]\n{r.text}"
            else:
                part = r.text
            
            if current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del índice RAG."""
        base_stats = self._pipeline.get_stats()
        return {
            **base_stats,
            "backend": self.backend_type,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }
    
    def clear(self) -> None:
        """Limpia todos los documentos indexados."""
        self._pipeline.documents = []
        self._pipeline.is_loaded = False
        if hasattr(self._pipeline, "collection") and self._pipeline.collection:
            # ChromaDB: recrear colección
            self._pipeline._setup_chromadb()
        logger.info("RAGEngine: índice limpiado")


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════

_engine_instance: Optional[RAGEngine] = None


def get_rag_engine(
    use_openai: Optional[bool] = None,
    singleton: bool = True,
) -> RAGEngine:
    """
    Obtiene una instancia del RAGEngine.
    
    Args:
        use_openai: Forzar backend (None = auto-detectar)
        singleton: Si True, retorna siempre la misma instancia
    
    Returns:
        Instancia de RAGEngine
    """
    global _engine_instance
    
    if singleton and _engine_instance is not None:
        return _engine_instance
    
    engine = RAGEngine(use_openai=use_openai)
    
    if singleton:
        _engine_instance = engine
    
    return engine


def reset_rag_engine() -> None:
    """Resetea el singleton (para tests)."""
    global _engine_instance
    _engine_instance = None
