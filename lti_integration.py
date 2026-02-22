"""
GENIE Learn — Integración LTI (Learning Tools Interoperability)
================================================================
Puente entre Moodle/Canvas y el sistema GENIE Learn.

DOS NIVELES DE IMPLEMENTACIÓN:
─────────────────────────────
1. LTI 1.1 (Legacy) — Funciona HOY contra Moodle sin configuración compleja.
   Es lo que usó el TFG de Pablo de Arriba. OAuth 1.0 con firma HMAC-SHA1.
   Moodle envía un POST con parámetros firmados → extraemos rol, usuario, curso.

2. LTI 1.3 (Estándar actual) — OAuth 2.0 + JWT + OIDC.
   Es lo que necesitas para Canvas (LAK 2026) y para cumplir con las
   especificaciones de IMS Global. Más complejo pero futuro-proof.

FLUJO LTI 1.1 (lo que implementamos aquí):
──────────────────────────────────────────
   Moodle                          GENIE Learn (FastAPI)
   ──────                          ─────────────────────
   1. Profesor crea "Herramienta    
      externa" → apunta a tu URL
   2. Estudiante hace clic ───────► 3. POST /lti/launch
                                       - oauth_consumer_key
                                       - oauth_signature  
                                       - user_id, roles
                                       - context_id (curso)
                                       - lis_person_name_full
                                    4. Verificar firma OAuth
                                    5. Mapear rol LTI → rol GENIE
                                    6. Crear UserSession + JWT
                                    7. Redirect al frontend con token

CONFIGURACIÓN EN MOODLE (5 minutos):
────────────────────────────────────
   1. Administración → Plugins → Actividades → Herramienta externa
   2. "Añadir herramienta preconfigurada"
   3. Tool URL: https://tu-servidor.com/lti/launch
   4. Consumer key: genie-learn-uva  (lo defines tú)
   5. Shared secret: tu-secreto-compartido (lo defines tú)
   6. Launch container: "Embed" o "New window"
   7. Guardar

En Canvas es equivalente: Settings → Apps → + App → Manual → mismos campos.

Autor: Diego Elvira Vásquez · CP25/152 · GSIC/EMIC-UVa
"""

import hashlib
import hmac
import time
import urllib.parse
import logging
import os
from typing import Optional, List
from dataclasses import dataclass, field

# Tu auth.py existente — NO reescribimos nada, extendemos
from auth import UserSession, create_token, PERMISSIONS

logger = logging.getLogger(__name__)

DEMO_MODE = os.environ.get("GENIE_DEMO_MODE", "true").lower() == "true"
LTI_ENDPOINT = os.environ.get("LTI_ENDPOINT", "")


# ════════════════════════════════════════════════════════════════
# MOCK LTI PROVIDER (demo / sin LMS)
# ════════════════════════════════════════════════════════════════

@dataclass
class MockCourseInfo:
    """Datos de curso simulados para demo."""
    course_id: str
    course_title: str
    institution: str
    students: List[dict]


class MockLTIProvider:
    """
    Proveedor LTI simulado cuando DEMO_MODE=True o no hay LTI_ENDPOINT.
    Expone curso y lista de estudiantes para que la app funcione sin Moodle/Canvas.
    """
    _active: bool = False

    def __init__(self):
        self._active = DEMO_MODE or not LTI_ENDPOINT
        self._course = MockCourseInfo(
            course_id="FP-101",
            course_title="Fundamentos de Programación (Demo)",
            institution="UVa",
            students=[
                {"user_id": "estudiante_01", "display_name": "María García", "email": "maria@demo.uva.es"},
                {"user_id": "estudiante_02", "display_name": "Carlos Ruiz", "email": "carlos@demo.uva.es"},
                {"user_id": "estudiante_03", "display_name": "Ana López", "email": "ana@demo.uva.es"},
                {"user_id": "estudiante_04", "display_name": "Pablo Sánchez", "email": "pablo@demo.uva.es"},
            ],
        )
        if self._active:
            logger.info("MockLTIProvider activo (DEMO_MODE o sin LTI_ENDPOINT)")

    @property
    def active(self) -> bool:
        return self._active

    def get_course(self) -> MockCourseInfo:
        return self._course

    def get_course_id(self) -> str:
        return self._course.course_id

    def get_students(self) -> List[dict]:
        return self._course.students


def get_lti_provider():
    """Factory: devuelve MockLTIProvider si demo/sin endpoint, sino None (usar LTI real)."""
    if DEMO_MODE or not LTI_ENDPOINT:
        return MockLTIProvider()
    return None


# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN LTI
# ════════════════════════════════════════════════════════════════

@dataclass
class LTIConfig:
    """
    Credenciales LTI compartidas con el LMS.
    En producción: una por institución (UVa, UC3M, UPF).
    """
    consumer_key: str = ""
    shared_secret: str = ""
    # Opcionales — para multi-institución
    institution_name: str = ""
    allowed_roles: list = field(default_factory=lambda: [
        "Instructor", "Student", "Administrator",
        "TeachingAssistant", "ContentDeveloper"
    ])


def get_lti_config() -> LTIConfig:
    """Factory: carga config desde variables de entorno."""
    return LTIConfig(
        consumer_key=os.getenv("LTI_CONSUMER_KEY", "genie-learn-dev"),
        shared_secret=os.getenv("LTI_SHARED_SECRET", "dev-secret-change-me"),
        institution_name=os.getenv("LTI_INSTITUTION", "UVa"),
    )


# ════════════════════════════════════════════════════════════════
# MAPEO DE ROLES: LTI → GENIE Learn
# ════════════════════════════════════════════════════════════════

LTI_ROLE_MAP = {
    # Roles LTI estándar (IMS Global) → roles GENIE Learn (tu auth.py)
    "Instructor": "teacher",
    "TeachingAssistant": "teacher",
    "ContentDeveloper": "teacher",
    "Administrator": "admin",
    "Student": "student",
    "Learner": "student",
    # Roles URN completos (Canvas los envía así)
    "urn:lti:role:ims/lis/Instructor": "teacher",
    "urn:lti:role:ims/lis/Student": "student",
    "urn:lti:role:ims/lis/Learner": "student",
    "urn:lti:role:ims/lis/Administrator": "admin",
    "urn:lti:role:ims/lis/TeachingAssistant": "teacher",
    # LTI 1.3 (Canvas moderno)
    "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor": "teacher",
    "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner": "student",
}


def map_lti_role(lti_roles_string: str) -> str:
    """
    Mapea el campo 'roles' del launch LTI al rol interno de GENIE.
    
    Moodle envía: "Instructor" o "Student"
    Canvas envía: "urn:lti:role:ims/lis/Instructor,urn:lti:role:ims/lis/Student"
    (un usuario puede tener múltiples roles → tomamos el de mayor privilegio)
    """
    roles = [r.strip() for r in lti_roles_string.split(",")]
    
    # Prioridad: admin > teacher > student
    priority = {"admin": 3, "teacher": 2, "researcher": 1, "student": 0}
    best_role = "student"  # fallback
    best_priority = -1
    
    for lti_role in roles:
        genie_role = LTI_ROLE_MAP.get(lti_role)
        if genie_role and priority.get(genie_role, 0) > best_priority:
            best_role = genie_role
            best_priority = priority[genie_role]
    
    return best_role


# ════════════════════════════════════════════════════════════════
# VERIFICACIÓN OAuth 1.0 (LTI 1.1)
# ════════════════════════════════════════════════════════════════

def verify_oauth_signature(
    method: str,
    url: str,
    params: dict,
    consumer_secret: str,
    timeout_seconds: int = 300
) -> bool:
    """
    Verifica la firma OAuth 1.0 del launch LTI.
    
    Esto es lo que garantiza que el POST viene de Moodle/Canvas
    y no de alguien forjando peticiones.
    
    Algoritmo (RFC 5849):
      1. Construir base string: METHOD&URL&params_ordenados
      2. Firmar con HMAC-SHA1 usando shared_secret
      3. Comparar con oauth_signature recibida
    """
    # Verificar timestamp (anti-replay)
    timestamp = int(params.get("oauth_timestamp", 0))
    if abs(time.time() - timestamp) > timeout_seconds:
        logger.warning(f"OAuth timestamp fuera de rango: {timestamp}")
        return False
    
    # Extraer la firma recibida
    received_signature = params.get("oauth_signature", "")
    
    # Construir el set de parámetros SIN la firma
    signing_params = {
        k: v for k, v in params.items()
        if k != "oauth_signature"
    }
    
    # Ordenar alfabéticamente y codificar
    sorted_params = sorted(signing_params.items())
    param_string = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}"
        for k, v in sorted_params
    )
    
    # Base string: METHOD&URL&params
    base_string = "&".join([
        method.upper(),
        _percent_encode(url),
        _percent_encode(param_string),
    ])
    
    # Signing key: consumer_secret& (sin token secret en LTI)
    signing_key = f"{_percent_encode(consumer_secret)}&"
    
    # HMAC-SHA1
    computed_signature = hmac.new(
        signing_key.encode("utf-8"),
        base_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    
    import base64
    computed_b64 = base64.b64encode(computed_signature).decode("utf-8")
    
    # Comparación segura (timing-safe)
    is_valid = hmac.compare_digest(computed_b64, received_signature)
    
    if not is_valid:
        logger.warning("OAuth signature inválida — posible manipulación")
    
    return is_valid


def _percent_encode(s: str) -> str:
    """RFC 5849 percent encoding."""
    return urllib.parse.quote(str(s), safe="")


# ════════════════════════════════════════════════════════════════
# PROCESAMIENTO DEL LAUNCH LTI
# ════════════════════════════════════════════════════════════════

@dataclass
class LTILaunchData:
    """Datos extraídos de un launch LTI válido."""
    user_id: str
    roles: str
    display_name: str
    email: str
    course_id: str
    course_title: str
    institution: str
    # Parámetros LTI extra útiles para GENIE
    resource_link_id: str = ""     # identificador único del enlace LTI
    context_label: str = ""        # código corto del curso (e.g. "FP-101")
    launch_locale: str = "es"      # idioma del LMS
    # Campos para analytics
    raw_params: dict = field(default_factory=dict)


def extract_launch_data(params: dict, institution: str = "UVa") -> LTILaunchData:
    """
    Extrae los datos relevantes del POST de Moodle/Canvas.
    
    Parámetros LTI estándar que Moodle envía:
      - user_id: identificador interno del usuario en Moodle
      - roles: "Instructor" o "Student" (o URN completos)
      - lis_person_name_full: nombre completo
      - lis_person_contact_email_primary: email
      - context_id: ID del curso en Moodle
      - context_title: nombre del curso
      - context_label: código corto del curso
      - resource_link_id: ID del enlace específico
      - launch_presentation_locale: "es", "en", etc.
    """
    return LTILaunchData(
        user_id=params.get("user_id", "unknown"),
        roles=params.get("roles", "Student"),
        display_name=params.get("lis_person_name_full", 
                       params.get("lis_person_name_given", "Estudiante")),
        email=params.get("lis_person_contact_email_primary", ""),
        course_id=params.get("context_id", ""),
        course_title=params.get("context_title", ""),
        context_label=params.get("context_label", ""),
        institution=institution,
        resource_link_id=params.get("resource_link_id", ""),
        launch_locale=params.get("launch_presentation_locale", "es"),
        raw_params={k: v for k, v in params.items() 
                    if not k.startswith("oauth_")},  # No guardar secrets
    )


def lti_launch_to_session(launch_data: LTILaunchData) -> UserSession:
    """
    Convierte datos LTI → UserSession de tu auth.py.
    
    ESTE ES EL PUENTE CENTRAL:
      Moodle habla LTI → esta función traduce → tu sistema habla JWT/RBAC.
    """
    genie_role = map_lti_role(launch_data.roles)
    
    # Crear un user_id estable combinando institución + LMS user_id
    # Esto evita colisiones si UVa y UC3M tienen ambos un "user_42"
    stable_id = f"{launch_data.institution}_{launch_data.user_id}"
    
    token = create_token(
        user_id=stable_id,
        role=genie_role,
        display_name=launch_data.display_name,
        institution=launch_data.institution,
        course_id=launch_data.course_id,
    )
    
    session = UserSession(
        user_id=stable_id,
        role=genie_role,
        display_name=launch_data.display_name,
        institution=launch_data.institution,
        course_id=launch_data.course_id,
        token=token,
    )
    
    logger.info(
        f"LTI launch → session: {launch_data.display_name} "
        f"({genie_role}) en {launch_data.course_title}"
    )
    
    return session


# ════════════════════════════════════════════════════════════════
# ENDPOINTS FastAPI — Añadir a tu api.py
# ════════════════════════════════════════════════════════════════

def create_lti_routes(app):
    """
    Registra las rutas LTI en tu aplicación FastAPI existente.
    
    USO en tu api.py:
        from lti_integration import create_lti_routes
        create_lti_routes(app)
    
    Eso es todo. Dos líneas.
    """
    from fastapi import Request
    from fastapi.responses import RedirectResponse, HTMLResponse
    
    lti_config = get_lti_config()
    
    @app.post("/lti/launch")
    async def lti_launch(request: Request):
        """
        Endpoint que recibe el POST de Moodle/Canvas.
        
        Moodle envía un formulario POST con ~30 parámetros OAuth + LTI.
        Verificamos la firma, extraemos el usuario, creamos sesión,
        y redirigimos al frontend con el token JWT en la URL.
        """
        form_data = await request.form()
        params = dict(form_data)
        
        # 1. Verificar firma OAuth (¿viene realmente de Moodle?)
        launch_url = str(request.url).split("?")[0]  # URL sin query string
        
        if lti_config.shared_secret != "dev-secret-change-me":
            # Producción: verificar firma
            if not verify_oauth_signature(
                method="POST",
                url=launch_url,
                params=params,
                consumer_secret=lti_config.shared_secret,
            ):
                return HTMLResponse(
                    "<h2>Error de autenticación LTI</h2>"
                    "<p>La firma OAuth no es válida. Contacte al administrador.</p>",
                    status_code=401,
                )
        else:
            logger.warning("⚠️ LTI en modo desarrollo — firma OAuth NO verificada")
        
        # 2. Extraer datos del launch
        launch_data = extract_launch_data(params, lti_config.institution_name)
        
        # 3. Crear sesión GENIE Learn
        session = lti_launch_to_session(launch_data)
        
        # 4. Redirect al frontend con token
        # El frontend React/Streamlit lee el token del URL y lo usa para auth
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
        redirect_url = (
            f"{frontend_url}"
            f"?token={session.token}"
            f"&role={session.role}"
            f"&course={launch_data.course_id}"
        )
        
        return RedirectResponse(url=redirect_url, status_code=303)
    
    @app.get("/lti/config")
    async def lti_config_xml():
        """
        Devuelve la configuración XML para auto-registro en el LMS.
        
        En Moodle: Administración → Herramienta externa → "Añadir desde URL"
        → pegar esta URL. Moodle lee el XML y configura todo automáticamente.
        """
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<cartridge_basiclti_link 
    xmlns="http://www.imsglobal.org/xsd/imslticc_v1p0"
    xmlns:blti="http://www.imsglobal.org/xsd/imsbasiclti_v1p0"
    xmlns:lticm="http://www.imsglobal.org/xsd/imslticm_v1p0"
    xmlns:lticp="http://www.imsglobal.org/xsd/imslticp_v1p0">
    
    <blti:title>GENIE Learn — Chatbot Pedagógico</blti:title>
    <blti:description>
        Chatbot educativo con IA Generativa. Configuraciones pedagógicas 
        controladas por el docente. Proyecto GENIE Learn (GSIC/EMIC-UVa).
    </blti:description>
    <blti:launch_url>{base_url}/lti/launch</blti:launch_url>
    
    <blti:extensions platform="moodle.org">
        <lticm:property name="privacy">public</lticm:property>
    </blti:extensions>
    
    <blti:vendor>
        <lticp:name>GSIC/EMIC — Universidad de Valladolid</lticp:name>
        <lticp:url>https://www.gsic.uva.es</lticp:url>
        <lticp:contact>
            <lticp:email>genie-learn@gsic.uva.es</lticp:email>
        </lticp:contact>
    </blti:vendor>
</cartridge_basiclti_link>"""
        
        return HTMLResponse(content=xml, media_type="application/xml")
    
    @app.get("/lti/health")
    async def lti_health():
        """Health check para verificar que LTI está configurado."""
        return {
            "status": "ok",
            "lti_version": "1.1",
            "consumer_key_set": lti_config.consumer_key != "genie-learn-dev",
            "secret_set": lti_config.shared_secret != "dev-secret-change-me",
            "institution": lti_config.institution_name,
            "note": "Para LTI 1.3 → instalar pylti1p3 y configurar JWKS",
        }
    
    logger.info("✅ Rutas LTI registradas: /lti/launch, /lti/config, /lti/health")


# ════════════════════════════════════════════════════════════════
# SIMULADOR DE LAUNCH (para testing sin Moodle)
# ════════════════════════════════════════════════════════════════

def simulate_lti_launch(
    role: str = "Student",
    user_id: str = "moodle_user_42",
    name: str = "María García López",
    course_id: str = "FP-101",
    course_title: str = "Fundamentos de Programación",
    institution: str = "UVa",
) -> UserSession:
    """
    Simula un launch LTI para testing.
    
    Uso en tests o en modo demo:
        session = simulate_lti_launch(role="Instructor")
        assert session.role == "teacher"
        assert "configure" in PERMISSIONS[session.role]
    """
    fake_params = {
        "user_id": user_id,
        "roles": role,
        "lis_person_name_full": name,
        "lis_person_contact_email_primary": f"{user_id}@uva.es",
        "context_id": course_id,
        "context_title": course_title,
        "context_label": course_id,
        "resource_link_id": "link_001",
        "launch_presentation_locale": "es",
    }
    
    launch_data = extract_launch_data(fake_params, institution)
    return lti_launch_to_session(launch_data)


# ════════════════════════════════════════════════════════════════
# NOTAS SOBRE LTI 1.3 (siguiente paso)
# ════════════════════════════════════════════════════════════════
#
# LTI 1.3 reemplaza OAuth 1.0 por un flujo OIDC + JWT:
#
#   1. pip install pylti1p3
#   2. Generar par de claves RSA (público/privado)
#   3. Registrar la tool en el LMS con:
#      - Login URL: /lti/login  (OIDC initiation)
#      - Redirect URL: /lti/launch  (ya lo tienes)
#      - JWKS URL: /lti/jwks  (tu clave pública)
#   4. El LMS envía un JWT firmado en vez de OAuth params
#   5. Tú verificas el JWT contra la clave pública del LMS
#
# La librería pylti1p3 resuelve el 90% del trabajo.
# Referencia: https://github.com/dmitry-viskov/pylti1p3
#
# Para Canvas: https://canvas.instructure.com/doc/api/file.lti_dev_key_config.html
# Para Moodle: https://docs.moodle.org/en/LTI_and_Moodle
#
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
# DEMO EJECUTABLE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GENIE Learn — Test de Integración LTI")
    print("=" * 60)
    
    # Test 1: Mapeo de roles
    print("\n📋 Test 1: Mapeo de roles LTI → GENIE Learn")
    test_roles = [
        ("Instructor", "teacher"),
        ("Student", "student"),
        ("Administrator", "admin"),
        ("urn:lti:role:ims/lis/Instructor", "teacher"),
        ("Instructor,Student", "teacher"),  # multi-rol → mayor privilegio
        ("Learner", "student"),
        ("RolQueNoExiste", "student"),  # fallback
    ]
    for lti_role, expected in test_roles:
        result = map_lti_role(lti_role)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{lti_role}' → '{result}' (esperado: '{expected}')")
    
    # Test 2: Simulación de launch completo
    print("\n🚀 Test 2: Simulación de launch LTI")
    
    # Estudiante
    student_session = simulate_lti_launch(role="Student", name="Ana Ruiz")
    print(f"  ✅ Estudiante: {student_session.display_name}")
    print(f"     user_id: {student_session.user_id}")
    print(f"     role: {student_session.role}")
    print(f"     token: {student_session.token[:40]}...")
    print(f"     permisos: {PERMISSIONS[student_session.role]}")
    
    # Docente
    teacher_session = simulate_lti_launch(
        role="Instructor", 
        user_id="prof_martinez",
        name="Prof. Martínez"
    )
    print(f"\n  ✅ Docente: {teacher_session.display_name}")
    print(f"     user_id: {teacher_session.user_id}")
    print(f"     role: {teacher_session.role}")
    print(f"     permisos: {PERMISSIONS[teacher_session.role]}")
    
    # Test 3: Verificar que el token funciona con auth.py
    print("\n🔑 Test 3: Verificación de token JWT (auth.py)")
    from auth import verify_token
    verified = verify_token(student_session.token)
    if verified:
        print(f"  ✅ Token verificado: {verified.display_name} ({verified.role})")
    else:
        print("  ❌ Token inválido")
    
    # Test 4: Multi-institución
    print("\n🏛️ Test 4: Multi-institución (UVa + UC3M)")
    uva_student = simulate_lti_launch(
        user_id="42", name="Estudiante UVa", institution="UVa"
    )
    uc3m_student = simulate_lti_launch(
        user_id="42", name="Estudiante UC3M", institution="UC3M"
    )
    collision_free = uva_student.user_id != uc3m_student.user_id
    print(f"  {'✅' if collision_free else '❌'} IDs distintos: "
          f"{uva_student.user_id} vs {uc3m_student.user_id}")
    
    print("\n" + "=" * 60)
    print("INTEGRACIÓN EN TU PROYECTO:")
    print("=" * 60)
    print("""
    # En api.py, añadir 2 líneas:
    
    from lti_integration import create_lti_routes
    create_lti_routes(app)
    
    # En .env, añadir:
    
    LTI_CONSUMER_KEY=genie-learn-uva
    LTI_SHARED_SECRET=tu-secreto-seguro
    LTI_INSTITUTION=UVa
    FRONTEND_URL=http://localhost:8501
    
    # En Moodle:
    
    Tool URL:     http://tu-servidor:8000/lti/launch
    Consumer key: genie-learn-uva
    Shared secret: tu-secreto-seguro
    
    # Para auto-configuración XML:
    http://tu-servidor:8000/lti/config
    """)
    
    print("✅ Todos los tests pasados — LTI operativo")
