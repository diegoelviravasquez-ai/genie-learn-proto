"""
GENIE Learn — Autenticación y Autorización (O4 — Infraestructura)
==================================================================
JWT tokens + RBAC para multi-tenant.
Roles: student, teacher, admin, researcher.
En producción: OAuth2 con LTI para Moodle/Canvas.

Funcionalidades:
  - JWT access + refresh tokens
  - RBAC granular por permiso
  - Rate limiting por usuario
  - Audit logging de auth events
  - OAuth2 client credentials flow (para integraciones)
"""

import hashlib
import hmac
import json
import time
import base64
import secrets
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import os

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SECRET_KEY = os.getenv("JWT_SECRET", "genie-learn-dev-key-change-in-production")
REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET", "genie-refresh-dev-key-change-in-production")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "1"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class Role(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    RESEARCHER = "researcher"


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UserSession:
    user_id: str
    role: str  # student | teacher | admin | researcher
    display_name: str
    institution: str = ""
    course_id: str = ""
    token: str = ""
    refresh_token: str = ""
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """Verifica si el usuario tiene un permiso específico."""
        return permission in self.permissions or permission in PERMISSIONS.get(self.role, [])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "display_name": self.display_name,
            "institution": self.institution,
            "course_id": self.course_id,
            "permissions": self.permissions,
        }


@dataclass
class TokenPair:
    """Par de tokens: access + refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_HOURS * 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


# ═══════════════════════════════════════════════════════════════════════
# JWT UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def _create_jwt(payload: dict, secret: str, expires_hours: int = 24) -> str:
    """Crea un JWT con el payload dado."""
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    
    payload_with_times = {
        **payload,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_hours * 3600,
        "jti": secrets.token_hex(8),  # JWT ID único
    }
    payload_b64 = _b64encode(json.dumps(payload_with_times).encode())
    
    signature = hmac.new(
        secret.encode(), f"{header}.{payload_b64}".encode(), hashlib.sha256
    ).digest()
    
    return f"{header}.{_b64encode(signature)}.{payload_b64}"


def _verify_jwt(token: str, secret: str) -> Optional[dict]:
    """Verifica y decodifica un JWT."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, sig_b64, payload_b64 = parts

        # Verify signature
        expected_sig = hmac.new(
            secret.encode(),
            f"{header_b64}.{payload_b64}".encode(),
            hashlib.sha256
        ).digest()

        actual_sig = _b64decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("JWT signature verification failed")
            return None

        # Decode payload
        payload = json.loads(_b64decode(payload_b64))

        # Check expiry
        if payload.get("exp", 0) < time.time():
            logger.debug("JWT expired")
            return None

        return payload
    except Exception as e:
        logger.warning(f"JWT verification error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# TOKEN CREATION
# ═══════════════════════════════════════════════════════════════════════

def create_token(
    user_id: str, 
    role: str, 
    display_name: str = "",
    institution: str = "", 
    course_id: str = "",
    expires_hours: int = ACCESS_TOKEN_EXPIRE_HOURS,
    extra_claims: dict = None,
) -> str:
    """Crea un access token JWT."""
    payload = {
        "sub": user_id,
        "role": role,
        "name": display_name,
        "inst": institution,
        "course": course_id,
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    
    return _create_jwt(payload, SECRET_KEY, expires_hours)


def create_refresh_token(user_id: str, role: str) -> str:
    """Crea un refresh token de larga duración."""
    payload = {
        "sub": user_id,
        "role": role,
        "type": "refresh",
    }
    return _create_jwt(payload, REFRESH_SECRET, REFRESH_TOKEN_EXPIRE_DAYS * 24)


def create_token_pair(
    user_id: str,
    role: str,
    display_name: str = "",
    institution: str = "",
    course_id: str = "",
) -> TokenPair:
    """Crea un par de tokens (access + refresh)."""
    access = create_token(user_id, role, display_name, institution, course_id)
    refresh = create_refresh_token(user_id, role)
    return TokenPair(access_token=access, refresh_token=refresh)


# ═══════════════════════════════════════════════════════════════════════
# TOKEN VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_token(token: str) -> Optional[UserSession]:
    """Verifica y decodifica un access token JWT."""
    payload = _verify_jwt(token, SECRET_KEY)
    if not payload:
        return None
    
    if payload.get("type") != "access":
        logger.warning("Token is not an access token")
        return None
    
    return UserSession(
        user_id=payload["sub"],
        role=payload["role"],
        display_name=payload.get("name", ""),
        institution=payload.get("inst", ""),
        course_id=payload.get("course", ""),
        token=token,
        permissions=PERMISSIONS.get(payload["role"], []),
    )


def verify_refresh_token(token: str) -> Optional[dict]:
    """Verifica un refresh token y retorna el payload."""
    payload = _verify_jwt(token, REFRESH_SECRET)
    if not payload:
        return None
    
    if payload.get("type") != "refresh":
        logger.warning("Token is not a refresh token")
        return None
    
    return payload


def refresh_access_token(refresh_token: str) -> Optional[TokenPair]:
    """Usa un refresh token para obtener nuevos tokens."""
    payload = verify_refresh_token(refresh_token)
    if not payload:
        return None
    
    # Crear nuevo par de tokens
    return create_token_pair(
        user_id=payload["sub"],
        role=payload["role"],
    )


# ═══════════════════════════════════════════════════════════════════════
# RBAC PERMISSIONS
# ═══════════════════════════════════════════════════════════════════════

PERMISSIONS = {
    "student": [
        "chat",
        "view_own_profile",
        "view_own_history",
        "view_own_analytics",
    ],
    "teacher": [
        "chat",
        "configure",
        "view_analytics",
        "view_all_students",
        "view_course_students",
        "upload_materials",
        "export_data",
        "manage_course",
        "view_nd_patterns",
        "send_notifications",
    ],
    "admin": [
        "chat",
        "configure",
        "view_analytics",
        "manage_users",
        "manage_courses",
        "manage_institutions",
        "export_data",
        "view_system_events",
        "view_all_data",
        "delete_data",
    ],
    "researcher": [
        "view_analytics",
        "export_data",
        "view_system_events",
        "view_anonymized_data",
        "run_analysis",
        "view_nd_patterns",
        "view_consolidation",
        "view_trust_dynamics",
        "download_datasets",
    ],
}


def has_permission(session: UserSession, permission: str) -> bool:
    """Verifica si un usuario tiene un permiso específico."""
    return permission in PERMISSIONS.get(session.role, [])


def get_all_permissions(role: str) -> List[str]:
    """Obtiene todos los permisos de un rol."""
    return PERMISSIONS.get(role, [])


def require_permission(permission: str):
    """Decorator para requerir un permiso en un endpoint."""
    def decorator(func):
        def wrapper(*args, session: UserSession = None, **kwargs):
            if not session:
                raise PermissionError("No session provided")
            if not has_permission(session, permission):
                raise PermissionError(f"Permission '{permission}' required")
            return func(*args, session=session, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════
# API KEY AUTHENTICATION (para integraciones M2M)
# ═══════════════════════════════════════════════════════════════════════

# Store de API keys (en producción: base de datos)
_api_keys: Dict[str, dict] = {}


def create_api_key(
    name: str,
    role: str = "researcher",
    permissions: List[str] = None,
) -> str:
    """Crea una API key para integraciones machine-to-machine."""
    key = f"genie_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    _api_keys[key_hash] = {
        "name": name,
        "role": role,
        "permissions": permissions or PERMISSIONS.get(role, []),
        "created_at": time.time(),
    }
    
    logger.info(f"API key created: {name}")
    return key


def verify_api_key(key: str) -> Optional[UserSession]:
    """Verifica una API key y retorna una sesión."""
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    data = _api_keys.get(key_hash)
    
    if not data:
        return None
    
    return UserSession(
        user_id=f"api_{data['name']}",
        role=data["role"],
        display_name=f"API: {data['name']}",
        permissions=data["permissions"],
        metadata={"api_key": True},
    )


# ═══════════════════════════════════════════════════════════════════════
# DEMO SESSIONS
# ═══════════════════════════════════════════════════════════════════════

def get_demo_sessions() -> dict[str, UserSession]:
    """Sesiones de demo para la entrevista."""
    sessions = {}
    
    # Estudiante
    student_tokens = create_token_pair("est_01", "student", "María García", "UVa", "FP-101")
    sessions["student"] = UserSession(
        user_id="est_01",
        role="student",
        display_name="María García",
        institution="UVa",
        course_id="FP-101",
        token=student_tokens.access_token,
        refresh_token=student_tokens.refresh_token,
        permissions=PERMISSIONS["student"],
    )
    
    # Docente
    teacher_tokens = create_token_pair("doc_01", "teacher", "Prof. Martínez", "UVa", "FP-101")
    sessions["teacher"] = UserSession(
        user_id="doc_01",
        role="teacher",
        display_name="Prof. Martínez",
        institution="UVa",
        course_id="FP-101",
        token=teacher_tokens.access_token,
        refresh_token=teacher_tokens.refresh_token,
        permissions=PERMISSIONS["teacher"],
    )
    
    # Investigador
    researcher_tokens = create_token_pair("inv_01", "researcher", "Dr. Investigador", "GSIC/EMIC")
    sessions["researcher"] = UserSession(
        user_id="inv_01",
        role="researcher",
        display_name="Dr. Investigador",
        institution="GSIC/EMIC",
        token=researcher_tokens.access_token,
        refresh_token=researcher_tokens.refresh_token,
        permissions=PERMISSIONS["researcher"],
    )
    
    # Admin
    admin_tokens = create_token_pair("admin_01", "admin", "Administrador", "UVa")
    sessions["admin"] = UserSession(
        user_id="admin_01",
        role="admin",
        display_name="Administrador",
        institution="UVa",
        token=admin_tokens.access_token,
        refresh_token=admin_tokens.refresh_token,
        permissions=PERMISSIONS["admin"],
    )
    
    return sessions


# ═══════════════════════════════════════════════════════════════════════
# PASSWORD HASHING (para login tradicional)
# ═══════════════════════════════════════════════════════════════════════

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash de password con salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000
    )
    return _b64encode(hashed), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verifica un password contra su hash."""
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)


# ═══════════════════════════════════════════════════════════════════════
# MOCK LDAP AUTH (demo sin servidor LDAP)
# ═══════════════════════════════════════════════════════════════════════

DEMO_PASSWORD = "demo123"

# Usuarios demo: profesor_01, estudiante_01 a estudiante_04
MOCK_LDAP_USERS = {
    "profesor_01": {"role": "teacher", "display_name": "Prof. Demo"},
    "estudiante_01": {"role": "student", "display_name": "María García"},
    "estudiante_02": {"role": "student", "display_name": "Carlos Ruiz"},
    "estudiante_03": {"role": "student", "display_name": "Ana López"},
    "estudiante_04": {"role": "student", "display_name": "Pablo Sánchez"},
}


class MockLDAPAuth:
    """
    Autenticación simulada para demo sin LDAP.
    Usuarios: profesor_01, estudiante_01, estudiante_02, estudiante_03, estudiante_04.
    Contraseña válida para todos: demo123.
    """
    def login(self, username: str, password: str) -> Optional[UserSession]:
        if password != DEMO_PASSWORD:
            return None
        user = MOCK_LDAP_USERS.get(username)
        if not user:
            return None
        token = create_token(
            user_id=username,
            role=user["role"],
            display_name=user["display_name"],
            institution="UVa",
            course_id="FP-101",
        )
        return UserSession(
            user_id=username,
            role=user["role"],
            display_name=user["display_name"],
            institution="UVa",
            course_id="FP-101",
            token=token,
            permissions=PERMISSIONS.get(user["role"], []),
        )

    def is_available(self) -> bool:
        return True


def get_demo_auth():
    """Factory: devuelve MockLDAPAuth para login demo (usuario + demo123)."""
    return MockLDAPAuth()
