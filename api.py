"""
GENIE Learn — API REST (FastAPI)
=================================
Backend que expone todos los servicios del chatbot pedagógico.
Capas: Auth → Rate Limiting → Validation → Middleware → RAG → LLM → Analytics → Response.

Endpoints:
  POST /api/chat           → Interacción estudiante-chatbot
  GET  /api/config         → Configuración pedagógica activa
  PUT  /api/config         → Actualizar configuración (docente)
  POST /api/ingest         → Ingestar documentos RAG
  GET  /api/analytics      → Dashboard analytics
  GET  /api/student/{id}   → Perfil cognitivo de estudiante
  GET  /api/health         → Health check
  POST /api/auth/login     → Login (demo)
  GET  /api/export/csv     → Export datos para investigación
  GET  /api/system-events  → Eventos sistema (investigador)

Seguridad:
  - JWT + RBAC (auth.py)
  - CORS configurable desde ALLOWED_ORIGINS env
  - Rate limiting en memoria por student_id (60 req/min)
  - Validación de input (max 2000 chars por mensaje)
  - Logging estructurado en todos los endpoints
  - try/except con HTTPException descriptivo en cada ruta
"""

import json
import logging
import os
import time
import tempfile
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator

from config import PedagogicalConfig, SystemConfig
from middleware import PedagogicalMiddleware
from rag_engine import RAGEngine, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client
from cognitive_engine import CognitiveEngine, BLOOM_LEVELS
from database import Database
from auth import (
    create_token, verify_token, has_permission,
    UserSession, get_demo_sessions
)

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("genie.api")

# ──────────────────────────────────────────────
# RATE LIMITER (en memoria — para producción usar Redis)
# ──────────────────────────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))


def check_rate_limit(user_id: str) -> None:
    """Lanza 429 si el usuario supera RATE_LIMIT_REQUESTS en RATE_LIMIT_WINDOW segundos."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = _rate_limit_store[user_id]
    # Limpiar llamadas fuera de ventana
    calls[:] = [t for t in calls if t > window_start]
    if len(calls) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit superado para usuario {user_id}: {len(calls)} req/{RATE_LIMIT_WINDOW}s")
        raise HTTPException(
            status_code=429,
            detail=f"Límite de {RATE_LIMIT_REQUESTS} solicitudes por minuto superado. Espera un momento."
        )
    calls.append(now)


# ──────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────

app = FastAPI(
    title="GENIE Learn API",
    description="Chatbot Pedagógico con IA Generativa — GSIC/EMIC UVa · CP25/152",
    version="0.2.0",
    docs_url="/docs",
)

# CORS: configurable desde entorno para no exponer * en producción
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if _allowed_origins_env == "*"
    else [o.strip() for o in _allowed_origins_env.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if allowed_origins == ["*"]:
    logger.warning("⚠️  CORS abierto a todos los orígenes (ALLOWED_ORIGINS=*). "
                   "Configura ALLOWED_ORIGINS en .env para producción.")

# ──────────────────────────────────────────────
# SERVICIOS GLOBALES
# ──────────────────────────────────────────────

try:
    config = PedagogicalConfig()
    middleware = PedagogicalMiddleware(config)
    rag = RAGEngine()
    llm = get_llm_client()
    cognitive = CognitiveEngine()
    db = Database()
    _n = rag.ingest_text(SAMPLE_COURSE_CONTENT, "Fundamentos_Programacion.pdf")
    db.ensure_user("doc_01", "teacher", "Prof. Martínez")
    db.ensure_course("FP-101", "Fundamentos de Programación", "doc_01")
    logger.info(f"Servicios inicializados — RAG: {_n} chunks, LLM: {llm.model_name}")
except Exception as e:
    logger.critical(f"Error inicializando servicios: {e}")
    raise

# ──────────────────────────────────────────────
# AUTH DEPENDENCY
# ──────────────────────────────────────────────

async def get_current_user(authorization: Optional[str] = Header(None)) -> UserSession:
    """Extrae y verifica el token JWT. Sin token → sesión demo de estudiante."""
    if not authorization:
        return get_demo_sessions()["student"]
    token = authorization.replace("Bearer ", "").strip()
    session = verify_token(token)
    if not session:
        logger.warning("Token inválido o expirado recibido")
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")
    return session

# ──────────────────────────────────────────────
# SCHEMAS CON VALIDACIÓN
# ──────────────────────────────────────────────

MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "2000"))


class ChatRequest(BaseModel):
    message: str
    student_id: Optional[str] = None
    session_id: Optional[str] = "default"

    @field_validator("message")
    @classmethod
    def message_not_empty_and_not_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El mensaje no puede estar vacío.")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Mensaje demasiado largo ({len(v)} chars). "
                f"Máximo permitido: {MAX_MESSAGE_LENGTH}."
            )
        return v


class ChatResponse(BaseModel):
    response: str
    scaffolding_level: int
    scaffolding_label: str
    bloom_level: int
    bloom_name: str
    icap_level: str
    trust_signal: str
    detected_topics: list
    copy_paste_score: float
    rag_sources: list
    remaining_prompts: int
    response_time_ms: int
    hallucination_injected: bool
    was_blocked: bool
    block_reason: str
    model_used: str


class ConfigUpdate(BaseModel):
    max_daily_prompts: Optional[int] = None
    scaffolding_mode: Optional[str] = None
    block_direct_solutions: Optional[bool] = None
    forced_hallucination_pct: Optional[float] = None
    use_rag: Optional[bool] = None
    no_context_behavior: Optional[str] = None
    model_name: Optional[str] = None
    role_play: Optional[str] = None
    system_addon: Optional[str] = None
    min_response_length: Optional[int] = None
    max_response_length: Optional[int] = None


class LoginRequest(BaseModel):
    user_id: str
    role: str = "student"
    display_name: str = ""

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        valid = {"student", "teacher", "researcher", "admin"}
        if v not in valid:
            raise ValueError(f"Rol inválido '{v}'. Roles válidos: {valid}")
        return v


SCAFFOLDING_LABELS = {0: "Socrático 🤔", 1: "Pista 💡", 2: "Ejemplo 📝", 3: "Explicación ✅"}

# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check — no requiere autenticación."""
    try:
        return {
            "status": "ok",
            "model": llm.model_name,
            "rag_chunks": rag.get_stats()["total_chunks"],
            "cors_origins": allowed_origins,
            "rate_limit": f"{RATE_LIMIT_REQUESTS}req/{RATE_LIMIT_WINDOW}s",
        }
    except Exception as e:
        logger.error(f"Health check fallido: {e}")
        raise HTTPException(status_code=503, detail="Servicio no disponible.")


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Login simplificado para demo. En producción: OAuth2 + LTI."""
    try:
        token = create_token(req.user_id, req.role, req.display_name, "UVa", config.course_id)
        db.ensure_user(req.user_id, req.role, req.display_name)
        logger.info(f"Login: {req.user_id} [{req.role}]")
        return {"token": token, "user_id": req.user_id, "role": req.role}
    except Exception as e:
        logger.error(f"Error en login para {req.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error procesando login.")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: UserSession = Depends(get_current_user)):
    """Endpoint principal del chatbot pedagógico."""
    student_id = req.student_id or user.user_id
    start_time = time.time()

    # Rate limiting
    check_rate_limit(student_id)

    logger.info(f"Chat — student={student_id}, msg_len={len(req.message)}, session={req.session_id}")

    try:
        # 1. PRE-PROCESO (middleware pedagógico)
        pre = middleware.pre_process(student_id, req.message)

        if not pre["allowed"]:
            logger.info(f"Prompt bloqueado — student={student_id}, reason={pre['block_reason']}")
            return ChatResponse(
                response=f"⛔ {pre['block_reason']}",
                scaffolding_level=pre.get("scaffolding_level", 0),
                scaffolding_label=SCAFFOLDING_LABELS.get(pre.get("scaffolding_level", 0), ""),
                bloom_level=0, bloom_name="", icap_level="", trust_signal="blocked",
                detected_topics=pre.get("detected_topics", []),
                copy_paste_score=pre.get("copy_paste_score", 0),
                rag_sources=[], remaining_prompts=0,
                response_time_ms=0, hallucination_injected=False,
                was_blocked=True, block_reason=pre["block_reason"],
                model_used=llm.model_name,
            )

        # 2. RAG RETRIEVAL
        context = ""
        rag_chunks = []
        if config.use_rag:
            try:
                context, rag_chunks = rag.build_context(req.message, top_k=config.rag_top_k)
                logger.debug(f"RAG: {len(rag_chunks)} chunks recuperados")
            except Exception as e:
                logger.warning(f"RAG falló, continuando sin contexto: {e}")

        # 3. LLM
        try:
            llm_result = llm.chat(
                system_prompt=pre["system_prompt"],
                user_prompt=req.message,
                context=context,
            )
        except Exception as e:
            logger.error(f"LLM falló para student={student_id}: {e}")
            raise HTTPException(
                status_code=503,
                detail="El modelo de lenguaje no está disponible. Inténtalo de nuevo."
            )

        # 4. POST-PROCESO
        post = middleware.post_process(student_id, llm_result["response"])

        # 5. COGNITIVE ANALYSIS (O1/O3)
        cog = cognitive.analyze_prompt(req.message)
        trust = cognitive.analyze_trust(student_id, req.message)
        cognitive.track_student(student_id, cog)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # 6. LOG MIDDLEWARE
        try:
            middleware.log_interaction(
                student_id=student_id,
                prompt_raw=req.message,
                pre_result=pre,
                response_raw=llm_result["response"],
                post_result=post,
                response_time_ms=elapsed_ms,
                bloom_level=cog.bloom_level,
                bloom_name=cog.bloom_name,
                trust_signal=trust.signal_type,
            )
        except Exception as e:
            logger.warning(f"Log middleware falló (no crítico): {e}")

        # 7. PERSISTENCIA DB (O4)
        try:
            db.log_interaction({
                "course_id": config.course_id,
                "student_id": student_id,
                "session_id": req.session_id,
                "prompt_raw": req.message,
                "prompt_processed": pre.get("processed_prompt", req.message),
                "detected_topics": json.dumps(pre.get("detected_topics", [])),
                "copy_paste_score": pre.get("copy_paste_score", 0),
                "was_blocked": 0,
                "block_reason": "",
                "scaffolding_level": pre.get("scaffolding_level", 0),
                "scaffolding_mode": config.scaffolding_mode,
                "response_raw": llm_result["response"],
                "response_delivered": post["response"],
                "model_used": llm_result.get("model", llm.model_name),
                "tokens_used": llm_result.get("tokens_used", 0),
                "response_time_ms": elapsed_ms,
                "hallucination_injected": 1 if post.get("hallucination_injected") else 0,
                "was_truncated": 1 if post.get("was_truncated") else 0,
                "bloom_level": cog.bloom_level,
                "bloom_name": cog.bloom_name,
                "icap_level": cog.icap_level,
                "engagement_score": cog.engagement_score,
                "trust_direction": trust.trust_direction,
                "trust_signal_type": trust.signal_type,
                "rag_chunks_used": len(rag_chunks),
                "rag_sources": json.dumps([c.get("source", "") for c in rag_chunks]),
                "rag_avg_score": round(
                    sum(c.get("score", 0) for c in rag_chunks) / max(len(rag_chunks), 1), 3
                ),
            })
            db.log_event("interaction", config.course_id, student_id, {
                "bloom": cog.bloom_level,
                "scaffolding": pre.get("scaffolding_level", 0),
                "trust": trust.signal_type,
                "elapsed_ms": elapsed_ms,
            })
        except Exception as e:
            logger.warning(f"Persistencia DB falló (no crítico): {e}")

        logger.info(
            f"Chat OK — student={student_id}, bloom={cog.bloom_level}, "
            f"trust={trust.signal_type}, {elapsed_ms}ms"
        )

        return ChatResponse(
            response=post["response"],
            scaffolding_level=pre.get("scaffolding_level", 0),
            scaffolding_label=SCAFFOLDING_LABELS.get(pre.get("scaffolding_level", 0), ""),
            bloom_level=cog.bloom_level,
            bloom_name=cog.bloom_name,
            icap_level=cog.icap_label,
            trust_signal=trust.signal_type,
            detected_topics=pre.get("detected_topics", []),
            copy_paste_score=pre.get("copy_paste_score", 0),
            rag_sources=[
                {"source": c["source"], "score": c["score"], "preview": c["text"][:120]}
                for c in rag_chunks
            ],
            remaining_prompts=middleware.get_remaining_prompts(student_id),
            response_time_ms=elapsed_ms,
            hallucination_injected=post.get("hallucination_injected", False),
            was_blocked=False,
            block_reason="",
            model_used=llm_result.get("model", llm.model_name),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en /api/chat para {student_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@app.get("/api/config")
async def get_config(user: UserSession = Depends(get_current_user)):
    try:
        return config.to_dict()
    except Exception as e:
        logger.error(f"Error obteniendo config: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo configuración.")


@app.put("/api/config")
async def update_config(update: ConfigUpdate, user: UserSession = Depends(get_current_user)):
    """Actualiza configuración pedagógica (solo docentes y admin)."""
    if not has_permission(user, "configure"):
        logger.warning(f"Acceso denegado a /api/config PUT — user={user.user_id} role={user.role}")
        raise HTTPException(status_code=403, detail="Sin permiso para modificar la configuración.")
    try:
        global config, middleware
        changed = {}
        for field_name, value in update.model_dump(exclude_none=True).items():
            if hasattr(config, field_name):
                setattr(config, field_name, value)
                changed[field_name] = value
        middleware.update_config(config)
        db.save_config(config.course_id, user.user_id, config.to_json(), "API update")
        db.log_event("config_change", config.course_id, user.user_id, changed)
        logger.info(f"Config actualizada por {user.user_id}: {changed}")
        return {"status": "ok", "config": config.to_dict(), "changed": changed}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando config: {e}")
        raise HTTPException(status_code=500, detail="Error actualizando configuración.")


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    user: UserSession = Depends(get_current_user)
):
    """Ingesta documento para RAG (solo teacher/admin)."""
    if not has_permission(user, "upload_materials"):
        logger.warning(f"Acceso denegado a /api/ingest — user={user.user_id}")
        raise HTTPException(status_code=403, detail="Sin permiso para subir materiales.")

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx. 50 MB).")

        if file.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(content)
                temp_path = f.name
            try:
                n = rag.ingest_pdf(temp_path)
            finally:
                os.unlink(temp_path)
        elif file.filename.endswith((".txt", ".md")):
            text = content.decode("utf-8")
            n = rag.ingest_text(text, file.filename)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado: {file.filename}. Use PDF, TXT o MD."
            )

        db.log_event("document_ingested", config.course_id, user.user_id,
                     {"filename": file.filename, "chunks": n, "size_bytes": len(content)})
        logger.info(f"Documento ingestado: {file.filename} → {n} chunks por {user.user_id}")
        return {"status": "ok", "filename": file.filename, "chunks_created": n, "stats": rag.get_stats()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingestando {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando el documento: {str(e)}")


@app.get("/api/analytics")
async def get_analytics(user: UserSession = Depends(get_current_user)):
    """Dashboard de analytics (O2/O3) — teacher, researcher, admin."""
    if not has_permission(user, "view_analytics"):
        raise HTTPException(status_code=403, detail="Sin permiso para ver analytics.")
    try:
        return {
            "middleware_summary": middleware.get_analytics_summary(),
            "db_summary": db.get_analytics_summary(config.course_id),
            "rag_stats": rag.get_stats(),
            "bloom_taxonomy": BLOOM_LEVELS,
            "config_history": db.get_config_history(config.course_id),
        }
    except Exception as e:
        logger.error(f"Error en analytics: {e}")
        raise HTTPException(status_code=500, detail="Error generando analytics.")


@app.get("/api/student/{student_id}")
async def get_student_profile(
    student_id: str,
    user: UserSession = Depends(get_current_user)
):
    """Perfil cognitivo de un estudiante (O1/O3)."""
    # Estudiante solo puede ver su propio perfil
    if not has_permission(user, "view_analytics") and user.user_id != student_id:
        raise HTTPException(status_code=403, detail="Solo puedes ver tu propio perfil.")
    try:
        profile = cognitive.get_student_profile(student_id)
        interactions = db.get_interactions(student_id=student_id, limit=20)
        return {"student_id": student_id, "profile": profile, "recent_interactions": interactions}
    except Exception as e:
        logger.error(f"Error obteniendo perfil de {student_id}: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo perfil del estudiante.")


@app.get("/api/export/csv")
async def export_csv(user: UserSession = Depends(get_current_user)):
    """Export datos para investigación (O1) — researcher, teacher, admin."""
    if not has_permission(user, "export_data"):
        raise HTTPException(status_code=403, detail="Sin permiso para exportar datos.")
    try:
        csv_data = db.export_interactions_csv(config.course_id)
        if not csv_data:
            return PlainTextResponse("No hay datos para exportar.", media_type="text/plain")
        logger.info(f"CSV exportado por {user.user_id} [{user.role}]")
        return PlainTextResponse(
            csv_data, media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=genie_export.csv"}
        )
    except Exception as e:
        logger.error(f"Error exportando CSV: {e}")
        raise HTTPException(status_code=500, detail="Error generando el export.")


@app.get("/api/system-events")
async def get_system_events(user: UserSession = Depends(get_current_user)):
    """Eventos del sistema para investigadores (O1)."""
    if not has_permission(user, "view_system_events"):
        raise HTTPException(status_code=403, detail="Sin permiso para ver eventos del sistema.")
    try:
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM system_events ORDER BY timestamp DESC LIMIT 200"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error obteniendo system events: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo eventos del sistema.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("ENV", "development") == "development"
    logger.info(f"Iniciando GENIE Learn API en puerto {port} (reload={reload})")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=reload)
