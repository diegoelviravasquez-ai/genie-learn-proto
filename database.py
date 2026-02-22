"""
GENIE Learn — Capa de Persistencia (O4 — Infraestructura)
==========================================================
SQLite para prototipo, migrable a PostgreSQL para producción.
Almacena: interacciones, configuraciones, usuarios, analytics.

Esquema diseñado para soportar:
  - Multi-tenant (course_id como partición lógica)
  - Audit trail completo (O1 — marco teórico necesita datos)
  - Export para análisis estadístico (O3 — pilotos WP5)
  - Migrations automáticas con versionado

MIGRATIONS:
  Cada migración tiene un número de versión y un SQL de upgrade.
  La tabla _migrations registra qué migraciones se han aplicado.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# MIGRATIONS
# ═══════════════════════════════════════════════════════════════════════

MIGRATIONS = [
    # V1: Schema inicial
    (1, """
        -- Usuarios (O4: multi-tenant)
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL CHECK(role IN ('student','teacher','admin','researcher')),
            display_name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            password_salt TEXT,
            institution TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            last_active TEXT,
            is_active INTEGER DEFAULT 1,
            metadata_json TEXT DEFAULT '{}'
        );

        -- Cursos
        CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            institution TEXT,
            teacher_id TEXT REFERENCES users(id),
            config_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 1
        );

        -- Enrollments (estudiantes en cursos)
        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL REFERENCES users(id),
            course_id TEXT NOT NULL REFERENCES courses(id),
            role TEXT DEFAULT 'student',
            enrolled_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, course_id)
        );

        -- Sesiones de chat
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            course_id TEXT REFERENCES courses(id),
            started_at TEXT DEFAULT (datetime('now')),
            ended_at TEXT,
            interaction_count INTEGER DEFAULT 0,
            metadata_json TEXT DEFAULT '{}'
        );

        -- Interacciones (el corazón del sistema - O1/O3)
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id TEXT REFERENCES courses(id),
            student_id TEXT NOT NULL,
            session_id TEXT REFERENCES sessions(id),
            timestamp TEXT DEFAULT (datetime('now')),

            -- Prompt
            prompt_raw TEXT NOT NULL,
            prompt_processed TEXT,
            detected_topics TEXT DEFAULT '[]',
            copy_paste_score REAL DEFAULT 0.0,

            -- Middleware decisions
            was_blocked INTEGER DEFAULT 0,
            block_reason TEXT DEFAULT '',
            scaffolding_level INTEGER DEFAULT 0,
            scaffolding_mode TEXT DEFAULT 'socratic',

            -- LLM response
            response_raw TEXT,
            response_delivered TEXT,
            model_used TEXT,
            tokens_used INTEGER DEFAULT 0,
            response_time_ms INTEGER DEFAULT 0,

            -- Post-processing
            hallucination_injected INTEGER DEFAULT 0,
            was_truncated INTEGER DEFAULT 0,

            -- Cognitive analysis (O1/O3)
            bloom_level INTEGER,
            bloom_name TEXT,
            icap_level TEXT,
            engagement_score REAL,

            -- Trust signals (O1)
            trust_direction REAL,
            trust_signal_type TEXT,

            -- RAG metadata
            rag_chunks_used INTEGER DEFAULT 0,
            rag_sources TEXT DEFAULT '[]',
            rag_avg_score REAL DEFAULT 0.0
        );

        -- Configuraciones históricas (O2: evolución de decisiones docentes)
        CREATE TABLE IF NOT EXISTS config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id TEXT REFERENCES courses(id),
            teacher_id TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            config_json TEXT NOT NULL,
            change_reason TEXT DEFAULT ''
        );

        -- Eventos del sistema (O1: logging semántico estructurado)
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            event_type TEXT NOT NULL,
            course_id TEXT,
            actor_id TEXT,
            payload TEXT DEFAULT '{}',
            severity TEXT DEFAULT 'info'
        );

        -- RAG documents metadata
        CREATE TABLE IF NOT EXISTS rag_documents (
            id TEXT PRIMARY KEY,
            course_id TEXT,
            filename TEXT,
            file_path TEXT,
            chunk_count INTEGER,
            ingested_at TEXT DEFAULT (datetime('now')),
            file_hash TEXT,
            metadata_json TEXT DEFAULT '{}'
        );

        -- Índices para queries frecuentes
        CREATE INDEX IF NOT EXISTS idx_interactions_student
            ON interactions(student_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_interactions_course
            ON interactions(course_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_interactions_session
            ON interactions(session_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_bloom
            ON interactions(bloom_level);
        CREATE INDEX IF NOT EXISTS idx_events_type
            ON system_events(event_type, timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_enrollments_course
            ON enrollments(course_id);
    """),
    
    # V2: ND Patterns & Consolidation tracking
    (2, """
        -- Patrones neurodivergentes detectados
        CREATE TABLE IF NOT EXISTS nd_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            course_id TEXT,
            pattern_type TEXT NOT NULL,
            confidence REAL,
            detected_at TEXT DEFAULT (datetime('now')),
            evidence_json TEXT DEFAULT '{}',
            is_active INTEGER DEFAULT 1
        );

        -- Consolidación entre sesiones
        CREATE TABLE IF NOT EXISTS consolidation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            session_1_id TEXT,
            session_2_id TEXT,
            bloom_delta INTEGER,
            hours_between REAL,
            consolidation_type TEXT,
            detected_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_nd_patterns_student
            ON nd_patterns(student_id);
        CREATE INDEX IF NOT EXISTS idx_consolidation_student
            ON consolidation_events(student_id);
    """),
    
    # V3: Teacher agency tracking
    (3, """
        -- Cambios de configuración del docente con contexto
        CREATE TABLE IF NOT EXISTS teacher_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT NOT NULL,
            course_id TEXT,
            action_type TEXT NOT NULL,
            config_before TEXT,
            config_after TEXT,
            trigger_reason TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );

        -- Feedback del docente sobre recomendaciones del sistema
        CREATE TABLE IF NOT EXISTS teacher_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT NOT NULL,
            recommendation_id TEXT,
            accepted INTEGER,
            modified INTEGER,
            reason TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_teacher_actions
            ON teacher_actions(teacher_id, timestamp);
    """),
    
    # V4: Research exports
    (4, """
        -- Export jobs para investigadores
        CREATE TABLE IF NOT EXISTS export_jobs (
            id TEXT PRIMARY KEY,
            researcher_id TEXT NOT NULL,
            export_type TEXT NOT NULL,
            parameters_json TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            file_path TEXT,
            row_count INTEGER
        );

        -- Anonymization mappings (para right to be forgotten)
        CREATE TABLE IF NOT EXISTS anonymization_map (
            original_id TEXT PRIMARY KEY,
            anonymous_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
]


class Database:
    """Capa de persistencia SQLite con migrations."""

    def __init__(self, db_path: str = "genie_learn.db"):
        self.db_path = db_path
        self._run_migrations()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _run_migrations(self):
        """Ejecuta migrations pendientes."""
        with self._conn() as conn:
            # Crear tabla de migrations si no existe
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Obtener versión actual
            row = conn.execute("SELECT MAX(version) as v FROM _migrations").fetchone()
            current_version = row["v"] or 0
            
            # Aplicar migrations pendientes
            for version, sql in MIGRATIONS:
                if version > current_version:
                    logger.info(f"Applying migration v{version}...")
                    conn.executescript(sql)
                    conn.execute("INSERT INTO _migrations (version) VALUES (?)", (version,))
                    logger.info(f"Migration v{version} applied")

    def get_version(self) -> int:
        """Retorna la versión actual del schema."""
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(version) as v FROM _migrations").fetchone()
            return row["v"] or 0

    # ═══════════════════════════════════════════════════════════════════
    # CRUD: Users
    # ═══════════════════════════════════════════════════════════════════

    def create_user(self, user_id: str, role: str, display_name: str = "",
                    email: str = None, institution: str = "") -> bool:
        """Crea un nuevo usuario."""
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO users (id, role, display_name, email, institution)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, role, display_name, email, institution))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, user_id: str) -> Optional[dict]:
        """Obtiene un usuario por ID."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def update_user(self, user_id: str, **kwargs) -> bool:
        """Actualiza campos de un usuario."""
        allowed = {"display_name", "email", "institution", "is_active", "metadata_json"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]
        
        with self._conn() as conn:
            conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        return True

    def ensure_user(self, user_id: str, role: str = "student", name: str = ""):
        """Crea usuario si no existe, actualiza last_active si existe."""
        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (id, role, display_name) VALUES (?,?,?)",
                    (user_id, role, name)
                )
            conn.execute("UPDATE users SET last_active=datetime('now') WHERE id=?", (user_id,))

    # ═══════════════════════════════════════════════════════════════════
    # CRUD: Courses
    # ═══════════════════════════════════════════════════════════════════

    def create_course(self, course_id: str, name: str, teacher_id: str = None,
                      institution: str = "", description: str = "") -> bool:
        """Crea un nuevo curso."""
        try:
            with self._conn() as conn:
                if teacher_id:
                    # Asegurar que el teacher existe
                    self.ensure_user(teacher_id, "teacher", f"Teacher {teacher_id}")
                conn.execute("""
                    INSERT INTO courses (id, name, description, institution, teacher_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (course_id, name, description, institution, teacher_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_course(self, course_id: str) -> Optional[dict]:
        """Obtiene un curso por ID."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
            return dict(row) if row else None

    def ensure_course(self, course_id: str, name: str = "", teacher_id: str = ""):
        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM courses WHERE id=?", (course_id,)).fetchone()
            if not existing:
                # Asegurar que el teacher existe si se especifica
                if teacher_id:
                    teacher_exists = conn.execute("SELECT id FROM users WHERE id=?", (teacher_id,)).fetchone()
                    if not teacher_exists:
                        conn.execute(
                            "INSERT INTO users (id, role, display_name) VALUES (?,?,?)",
                            (teacher_id, "teacher", f"Teacher {teacher_id}")
                        )
                conn.execute(
                    "INSERT INTO courses (id, name, teacher_id) VALUES (?,?,?)",
                    (course_id, name, teacher_id if teacher_id else None)
                )

    def enroll_user(self, user_id: str, course_id: str, role: str = "student") -> bool:
        """Matricula un usuario en un curso."""
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO enrollments (user_id, course_id, role)
                    VALUES (?, ?, ?)
                """, (user_id, course_id, role))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_course_students(self, course_id: str) -> List[dict]:
        """Obtiene todos los estudiantes de un curso."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT u.* FROM users u
                JOIN enrollments e ON u.id = e.user_id
                WHERE e.course_id = ? AND e.role = 'student'
            """, (course_id,)).fetchall()
            return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    # CRUD: Sessions
    # ═══════════════════════════════════════════════════════════════════

    def create_session(self, session_id: str, user_id: str, course_id: str = None) -> bool:
        """Crea una nueva sesión de chat."""
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO sessions (id, user_id, course_id)
                    VALUES (?, ?, ?)
                """, (session_id, user_id, course_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def update_session_count(self, session_id: str) -> None:
        """Incrementa el contador de interacciones de una sesión."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE sessions 
                SET interaction_count = interaction_count + 1
                WHERE id = ?
            """, (session_id,))

    def end_session(self, session_id: str) -> None:
        """Marca una sesión como terminada."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE sessions SET ended_at = datetime('now') WHERE id = ?
            """, (session_id,))

    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[dict]:
        """Obtiene las sesiones recientes de un usuario."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM sessions 
                WHERE user_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
            return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    # CRUD: Interactions
    # ═══════════════════════════════════════════════════════════════════

    def log_interaction(self, data: dict) -> int:
        """Registra una interacción."""
        # Valores por defecto
        defaults = {
            'course_id': None, 'session_id': None, 'prompt_processed': None,
            'detected_topics': '[]', 'copy_paste_score': 0.0,
            'was_blocked': 0, 'block_reason': '', 'scaffolding_level': 0,
            'scaffolding_mode': 'socratic', 'response_raw': None,
            'response_delivered': None, 'model_used': None, 'tokens_used': 0,
            'response_time_ms': 0, 'hallucination_injected': 0, 'was_truncated': 0,
            'bloom_level': None, 'bloom_name': None, 'icap_level': None,
            'engagement_score': None, 'trust_direction': None,
            'trust_signal_type': None, 'rag_chunks_used': 0,
            'rag_sources': '[]', 'rag_avg_score': 0.0,
        }
        
        # Merge con datos proporcionados
        full_data = {**defaults, **data}
        
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO interactions (
                    course_id, student_id, session_id,
                    prompt_raw, prompt_processed, detected_topics, copy_paste_score,
                    was_blocked, block_reason, scaffolding_level, scaffolding_mode,
                    response_raw, response_delivered, model_used, tokens_used, response_time_ms,
                    hallucination_injected, was_truncated,
                    bloom_level, bloom_name, icap_level, engagement_score,
                    trust_direction, trust_signal_type,
                    rag_chunks_used, rag_sources, rag_avg_score
                ) VALUES (
                    :course_id, :student_id, :session_id,
                    :prompt_raw, :prompt_processed, :detected_topics, :copy_paste_score,
                    :was_blocked, :block_reason, :scaffolding_level, :scaffolding_mode,
                    :response_raw, :response_delivered, :model_used, :tokens_used, :response_time_ms,
                    :hallucination_injected, :was_truncated,
                    :bloom_level, :bloom_name, :icap_level, :engagement_score,
                    :trust_direction, :trust_signal_type,
                    :rag_chunks_used, :rag_sources, :rag_avg_score
                )
            """, full_data)
            
            # Actualizar contador de sesión si existe
            if full_data.get('session_id'):
                self.update_session_count(full_data['session_id'])
            
            return cur.lastrowid

    def get_interactions(self, course_id: str = None, student_id: str = None,
                         session_id: str = None, limit: int = 100, 
                         offset: int = 0) -> List[dict]:
        """Obtiene interacciones con filtros opcionales."""
        query = "SELECT * FROM interactions WHERE 1=1"
        params = {}
        
        if course_id:
            query += " AND course_id = :course_id"
            params["course_id"] = course_id
        if student_id:
            query += " AND student_id = :student_id"
            params["student_id"] = student_id
        if session_id:
            query += " AND session_id = :session_id"
            params["session_id"] = session_id
            
        query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_daily_count(self, student_id: str, date_str: str = None) -> int:
        """Cuenta interacciones de un estudiante en un día."""
        if not date_str:
            date_str = date.today().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM interactions WHERE student_id=? AND date(timestamp)=?",
                (student_id, date_str)
            ).fetchone()
            return row["cnt"] if row else 0

    def get_student_topics(self, student_id: str, days: int = 30) -> Dict[str, int]:
        """Obtiene los temas consultados por un estudiante."""
        with self._conn() as conn:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            rows = conn.execute("""
                SELECT detected_topics FROM interactions 
                WHERE student_id = ? AND timestamp > ?
            """, (student_id, since)).fetchall()
            
            topic_counts = {}
            for row in rows:
                try:
                    topics = json.loads(row["detected_topics"])
                    for t in topics:
                        topic_counts[t] = topic_counts.get(t, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            return topic_counts

    # ═══════════════════════════════════════════════════════════════════
    # System Events (O1)
    # ═══════════════════════════════════════════════════════════════════

    def log_event(self, event_type: str, course_id: str = "", actor_id: str = "",
                  payload: dict = None, severity: str = "info"):
        """Registra un evento del sistema."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO system_events (event_type, course_id, actor_id, payload, severity) 
                VALUES (?,?,?,?,?)
            """, (event_type, course_id, actor_id, json.dumps(payload or {}), severity))

    def get_events(self, event_type: str = None, course_id: str = None,
                   since: str = None, limit: int = 100) -> List[dict]:
        """Obtiene eventos del sistema."""
        query = "SELECT * FROM system_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if course_id:
            query += " AND course_id = ?"
            params.append(course_id)
        if since:
            query += " AND timestamp > ?"
            params.append(since)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    # Config History (O2)
    # ═══════════════════════════════════════════════════════════════════

    def save_config(self, course_id: str, teacher_id: str, config_json: str, reason: str = ""):
        """Guarda un snapshot de configuración."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO config_history (course_id, teacher_id, config_json, change_reason) 
                VALUES (?,?,?,?)
            """, (course_id, teacher_id, config_json, reason))

    def get_config_history(self, course_id: str, limit: int = 50) -> List[dict]:
        """Obtiene el historial de configuraciones."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM config_history WHERE course_id=? 
                ORDER BY timestamp DESC LIMIT ?
            """, (course_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_config(self, course_id: str) -> Optional[dict]:
        """Obtiene la configuración más reciente."""
        history = self.get_config_history(course_id, limit=1)
        if history:
            return json.loads(history[0]["config_json"])
        return None

    # ═══════════════════════════════════════════════════════════════════
    # ND Patterns
    # ═══════════════════════════════════════════════════════════════════

    def log_nd_pattern(self, student_id: str, pattern_type: str, confidence: float,
                       course_id: str = None, evidence: dict = None) -> int:
        """Registra un patrón neurodivergente detectado."""
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO nd_patterns (student_id, course_id, pattern_type, confidence, evidence_json)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, course_id, pattern_type, confidence, json.dumps(evidence or {})))
            return cur.lastrowid

    def get_student_nd_patterns(self, student_id: str) -> List[dict]:
        """Obtiene patrones ND de un estudiante."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM nd_patterns 
                WHERE student_id = ? AND is_active = 1
                ORDER BY detected_at DESC
            """, (student_id,)).fetchall()
            return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    # Consolidation Events
    # ═══════════════════════════════════════════════════════════════════

    def log_consolidation(self, student_id: str, topic: str, bloom_delta: int,
                          hours_between: float, consolidation_type: str,
                          session_1_id: str = None, session_2_id: str = None) -> int:
        """Registra un evento de consolidación."""
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO consolidation_events 
                (student_id, topic, session_1_id, session_2_id, bloom_delta, hours_between, consolidation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (student_id, topic, session_1_id, session_2_id, bloom_delta, hours_between, consolidation_type))
            return cur.lastrowid

    def get_student_consolidation(self, student_id: str, days: int = 30) -> List[dict]:
        """Obtiene eventos de consolidación de un estudiante."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM consolidation_events 
                WHERE student_id = ? AND detected_at > ?
                ORDER BY detected_at DESC
            """, (student_id, since)).fetchall()
            return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    # Analytics Aggregates
    # ═══════════════════════════════════════════════════════════════════

    def get_analytics_summary(self, course_id: str = None) -> dict:
        """Resumen de analytics para dashboard."""
        with self._conn() as conn:
            base_where = "WHERE course_id=?" if course_id else "WHERE 1=1"
            params = (course_id,) if course_id else ()

            total = conn.execute(f"SELECT COUNT(*) as c FROM interactions {base_where}", params).fetchone()["c"]
            if total == 0:
                return {"total_interactions": 0}

            students = conn.execute(
                f"SELECT COUNT(DISTINCT student_id) as c FROM interactions {base_where}", params
            ).fetchone()["c"]

            avg_bloom = conn.execute(
                f"SELECT AVG(bloom_level) as avg FROM interactions {base_where} AND bloom_level IS NOT NULL",
                params
            ).fetchone()["avg"]

            blocked = conn.execute(
                f"SELECT COUNT(*) as c FROM interactions {base_where} AND was_blocked=1", params
            ).fetchone()["c"]

            avg_time = conn.execute(
                f"SELECT AVG(response_time_ms) as avg FROM interactions {base_where}", params
            ).fetchone()["avg"]

            # Bloom distribution
            bloom_dist = {}
            for row in conn.execute(
                f"SELECT bloom_level, COUNT(*) as c FROM interactions {base_where} AND bloom_level IS NOT NULL GROUP BY bloom_level",
                params
            ).fetchall():
                bloom_dist[row["bloom_level"]] = row["c"]

            # Topic distribution
            topic_dist = {}
            for row in conn.execute(f"SELECT detected_topics FROM interactions {base_where}", params).fetchall():
                try:
                    topics = json.loads(row["detected_topics"])
                    for t in topics:
                        topic_dist[t] = topic_dist.get(t, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Scaffolding distribution
            scaffolding_dist = {}
            for row in conn.execute(
                f"SELECT scaffolding_mode, COUNT(*) as c FROM interactions {base_where} GROUP BY scaffolding_mode",
                params
            ).fetchall():
                scaffolding_dist[row["scaffolding_mode"]] = row["c"]

            return {
                "total_interactions": total,
                "unique_students": students,
                "avg_bloom_level": round(avg_bloom, 2) if avg_bloom else 0,
                "blocked_requests": blocked,
                "blocked_rate": round(blocked / total * 100, 1) if total > 0 else 0,
                "avg_response_time_ms": round(avg_time) if avg_time else 0,
                "bloom_distribution": bloom_dist,
                "topic_distribution": topic_dist,
                "scaffolding_distribution": scaffolding_dist,
            }

    # ═══════════════════════════════════════════════════════════════════
    # Export for Research (O1/O3)
    # ═══════════════════════════════════════════════════════════════════

    def export_interactions_csv(self, course_id: str = None, anonymize: bool = True) -> str:
        """Export all interactions as CSV for statistical analysis."""
        import csv
        import io
        import hashlib

        rows = self.get_interactions(course_id=course_id, limit=100000)
        if not rows:
            return ""

        output = io.StringIO()
        
        # Anonymize if requested
        if anonymize:
            id_map = {}
            for row in rows:
                for field in ['student_id', 'course_id', 'session_id']:
                    if row.get(field) and row[field] not in id_map:
                        id_map[row[field]] = hashlib.sha256(row[field].encode()).hexdigest()[:12]
                    if row.get(field):
                        row[field] = id_map.get(row[field], row[field])
        
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    def export_for_research(self, export_type: str, course_id: str = None,
                            researcher_id: str = None) -> Dict[str, Any]:
        """Export completo para investigación."""
        result = {
            "export_type": export_type,
            "course_id": course_id,
            "generated_at": datetime.now().isoformat(),
            "data": {},
        }
        
        if export_type in ("full", "interactions"):
            result["data"]["interactions"] = self.get_interactions(course_id=course_id, limit=100000)
        
        if export_type in ("full", "analytics"):
            result["data"]["analytics"] = self.get_analytics_summary(course_id)
        
        if export_type in ("full", "events"):
            result["data"]["events"] = self.get_events(course_id=course_id, limit=10000)
        
        if export_type in ("full", "config_history"):
            if course_id:
                result["data"]["config_history"] = self.get_config_history(course_id, limit=1000)
        
        # Log export event
        self.log_event(
            "data_export",
            course_id=course_id or "",
            actor_id=researcher_id or "",
            payload={"export_type": export_type, "row_counts": {k: len(v) if isinstance(v, list) else 1 for k, v in result["data"].items()}},
        )
        
        return result


# ═══════════════════════════════════════════════════════════════════════
# DEMO: SQLite local genie_demo.db con datos de ejemplo
# ═══════════════════════════════════════════════════════════════════════

DEMO_DB_PATH = "genie_demo.db"


def seed_demo_data(db: Database) -> None:
    """Inserta usuarios, curso y enrollments de ejemplo en la base demo."""
    with db._conn() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            return  # ya tiene datos
    db.ensure_user("profesor_01", "teacher", "Prof. Demo")
    for i, name in enumerate(["María García", "Carlos Ruiz", "Ana López", "Pablo Sánchez"], 1):
        uid = f"estudiante_0{i}"
        db.ensure_user(uid, "student", name)
    db.ensure_course("FP-101", "Fundamentos de Programación (Demo)", "profesor_01")
    for i in range(1, 5):
        db.enroll_user(f"estudiante_0{i}", "FP-101", "student")
    db.enroll_user("profesor_01", "FP-101", "teacher")
    logger.info("Demo data seeded in %s", db.db_path)


def get_demo_database() -> Database:
    """Devuelve una instancia de Database usando genie_demo.db y datos de ejemplo."""
    db = Database(db_path=DEMO_DB_PATH)
    seed_demo_data(db)
    return db
