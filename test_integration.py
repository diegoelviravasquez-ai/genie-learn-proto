"""
TEST DE INTEGRACIÓN — GENIE Learn Ecosystem
============================================
Tests end-to-end que verifican que todos los componentes
funcionan correctamente juntos.

Ejecutar: pytest test_integration.py -v
"""

import os
import sys
import tempfile
import pytest
from typing import Generator


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Crea una base de datos temporal."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def database(temp_db):
    """Inicializa la base de datos."""
    from database import Database
    return Database(temp_db)


@pytest.fixture
def middleware():
    """Middleware con configuración por defecto."""
    from middleware import PedagogicalMiddleware, PedagogicalConfig
    config = PedagogicalConfig(
        scaffolding_mode='socratic',
        max_daily_prompts=10,
        block_direct_solutions=True,
    )
    return PedagogicalMiddleware(config)


@pytest.fixture
def rag_engine():
    """RAG engine con contenido de demo."""
    from rag_engine import RAGEngine, SAMPLE_COURSE_CONTENT
    engine = RAGEngine(use_openai=False)
    engine.ingest_text(SAMPLE_COURSE_CONTENT, 'demo.pdf')
    return engine


@pytest.fixture
def llm_client():
    """Cliente LLM (mock mode)."""
    from llm_client import get_llm_client
    return get_llm_client()


@pytest.fixture
def cognitive_analyzer():
    """Analizador cognitivo."""
    from cognitive_analyzer import CognitiveAnalyzer
    return CognitiveAnalyzer()


# ═══════════════════════════════════════════════════════════════════════
# TESTS: CORE STACK
# ═══════════════════════════════════════════════════════════════════════

class TestCoreStack:
    """Tests del stack principal: middleware → RAG → LLM."""
    
    def test_middleware_preprocess(self, middleware):
        """Verifica que el middleware pre-procesa correctamente."""
        result = middleware.pre_process('student_01', '¿Qué es un bucle for?')
        
        assert result['allowed'] is True
        assert 'bucles' in result['detected_topics']
        assert result['scaffolding_level'] >= 0
        assert len(result['system_prompt']) > 0
    
    def test_middleware_blocks_limit(self, middleware):
        """Verifica que el middleware bloquea al superar límite diario."""
        student_id = 'student_limit_test'
        
        # Simular que ya llegó al límite
        for i in range(10):
            middleware.pre_process(student_id, f'Pregunta {i}')
        
        # La siguiente debe estar bloqueada
        result = middleware.pre_process(student_id, 'Una más')
        assert result['allowed'] is False
        assert 'límite' in result['block_reason'].lower() or 'limit' in result['block_reason'].lower()
    
    def test_rag_retrieval(self, rag_engine):
        """Verifica que el RAG recupera chunks relevantes."""
        results = rag_engine.retrieve('bucle for', top_k=3)
        
        assert len(results) > 0
        assert all(r.score >= 0 for r in results)
        assert any('for' in r.text.lower() or 'bucle' in r.text.lower() for r in results)
    
    def test_rag_build_context(self, rag_engine):
        """Verifica la construcción de contexto RAG."""
        context = rag_engine.build_context('funciones en Python', top_k=2)
        
        assert len(context) > 0
        assert 'Fragmento' in context  # Formato de citación
    
    def test_llm_chat(self, llm_client):
        """Verifica que el cliente LLM responde."""
        response = llm_client.chat(
            'Eres un asistente educativo.',
            '¿Qué es Python?',
            'Python es un lenguaje de programación.'
        )
        
        assert 'response' in response
        assert len(response['response']) > 0
    
    def test_full_flow(self, middleware, rag_engine, llm_client, cognitive_analyzer):
        """Test del flujo completo estudiante → respuesta."""
        student_id = 'test_student_flow'
        prompt = '¿Cómo puedo crear una lista en Python?'
        
        # 1. Pre-process
        pre = middleware.pre_process(student_id, prompt)
        assert pre['allowed'] is True
        
        # 2. RAG
        context = rag_engine.build_context(prompt)
        assert len(context) > 0
        
        # 3. LLM
        response = llm_client.chat(pre['system_prompt'], prompt, context)
        assert len(response['response']) > 0
        
        # 4. Post-process
        post = middleware.post_process(student_id, response['response'])
        assert 'response' in post
        
        # 5. Análisis cognitivo
        analysis = cognitive_analyzer.analyze(prompt)
        assert analysis.bloom_level >= 1


# ═══════════════════════════════════════════════════════════════════════
# TESTS: AUTH
# ═══════════════════════════════════════════════════════════════════════

class TestAuth:
    """Tests del sistema de autenticación."""
    
    def test_create_token(self):
        """Verifica creación de tokens."""
        from auth import create_token, verify_token
        
        token = create_token('user_01', 'teacher', 'Prof. Test')
        assert len(token) > 100
        
        session = verify_token(token)
        assert session is not None
        assert session.user_id == 'user_01'
        assert session.role == 'teacher'
    
    def test_token_pair(self):
        """Verifica creación de par access+refresh."""
        from auth import create_token_pair, verify_token, verify_refresh_token
        
        pair = create_token_pair('user_02', 'student')
        
        assert len(pair.access_token) > 0
        assert len(pair.refresh_token) > 0
        
        # Access token válido
        session = verify_token(pair.access_token)
        assert session.user_id == 'user_02'
        
        # Refresh token válido
        payload = verify_refresh_token(pair.refresh_token)
        assert payload['sub'] == 'user_02'
    
    def test_permissions(self):
        """Verifica sistema de permisos RBAC."""
        from auth import create_token, verify_token, has_permission
        
        # Teacher tiene permiso de configurar
        teacher_token = create_token('t1', 'teacher')
        teacher_session = verify_token(teacher_token)
        assert has_permission(teacher_session, 'configure') is True
        
        # Student NO tiene permiso de configurar
        student_token = create_token('s1', 'student')
        student_session = verify_token(student_token)
        assert has_permission(student_session, 'configure') is False
    
    def test_demo_sessions(self):
        """Verifica sesiones de demo."""
        from auth import get_demo_sessions
        
        sessions = get_demo_sessions()
        
        assert 'student' in sessions
        assert 'teacher' in sessions
        assert 'researcher' in sessions
        
        assert sessions['teacher'].role == 'teacher'


# ═══════════════════════════════════════════════════════════════════════
# TESTS: DATABASE
# ═══════════════════════════════════════════════════════════════════════

class TestDatabase:
    """Tests de la capa de persistencia."""
    
    def test_migrations(self, database):
        """Verifica que las migrations se aplican."""
        version = database.get_version()
        assert version >= 4  # Deberían estar las 4 migrations
    
    def test_user_crud(self, database):
        """Verifica CRUD de usuarios."""
        # Create
        created = database.create_user('u1', 'student', 'Test User', 'test@test.com')
        assert created is True
        
        # Read
        user = database.get_user('u1')
        assert user is not None
        assert user['display_name'] == 'Test User'
        
        # Update
        database.update_user('u1', display_name='Updated Name')
        user = database.get_user('u1')
        assert user['display_name'] == 'Updated Name'
        
        # Duplicate fails
        created_again = database.create_user('u1', 'student')
        assert created_again is False
    
    def test_course_crud(self, database):
        """Verifica CRUD de cursos."""
        # Create user primero
        database.create_user('teacher_01', 'teacher', 'Prof')
        
        # Create course
        created = database.create_course('C1', 'Test Course', 'teacher_01')
        assert created is True
        
        # Read
        course = database.get_course('C1')
        assert course is not None
        assert course['name'] == 'Test Course'
    
    def test_log_interaction(self, database):
        """Verifica logging de interacciones."""
        database.ensure_user('s1', 'student')
        database.ensure_course('C1', 'Course')
        
        interaction_id = database.log_interaction({
            'student_id': 's1',
            'course_id': 'C1',
            'prompt_raw': '¿Qué es Python?',
            'bloom_level': 2,
        })
        
        assert interaction_id > 0
        
        interactions = database.get_interactions(student_id='s1')
        assert len(interactions) == 1
        assert interactions[0]['prompt_raw'] == '¿Qué es Python?'
    
    def test_analytics_summary(self, database):
        """Verifica resumen de analytics."""
        database.ensure_user('s1', 'student')
        database.ensure_course('C1', 'Course')
        
        # Log varias interacciones
        for i in range(5):
            database.log_interaction({
                'student_id': 's1',
                'course_id': 'C1',
                'prompt_raw': f'Pregunta {i}',
                'bloom_level': i % 6 + 1,
            })
        
        summary = database.get_analytics_summary('C1')
        
        assert summary['total_interactions'] == 5
        assert summary['unique_students'] == 1
        assert 'bloom_distribution' in summary
    
    def test_nd_patterns(self, database):
        """Verifica logging de patrones ND."""
        database.ensure_user('s1', 'student')
        
        pattern_id = database.log_nd_pattern(
            's1', 'EPISODIC', 0.8,
            evidence={'events': ['topic_switch', 'topic_switch']}
        )
        
        assert pattern_id > 0
        
        patterns = database.get_student_nd_patterns('s1')
        assert len(patterns) == 1
        assert patterns[0]['pattern_type'] == 'EPISODIC'
    
    def test_consolidation(self, database):
        """Verifica logging de consolidación."""
        database.ensure_user('s1', 'student')
        
        cons_id = database.log_consolidation(
            's1', 'bucles', bloom_delta=2,
            hours_between=48.0, consolidation_type='positive'
        )
        
        assert cons_id > 0
        
        events = database.get_student_consolidation('s1')
        assert len(events) == 1
        assert events[0]['bloom_delta'] == 2


# ═══════════════════════════════════════════════════════════════════════
# TESTS: COGNITIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

class TestCognitiveAnalysis:
    """Tests del análisis cognitivo."""
    
    def test_bloom_detection(self, cognitive_analyzer):
        """Verifica detección de niveles Bloom."""
        from cognitive_analyzer import BLOOM_LEVELS
        
        # Nivel 1 - Recordar
        analysis = cognitive_analyzer.analyze('¿Qué es una variable?')
        assert analysis.bloom_level <= 2
        
        # Nivel 4+ - Analizar
        analysis = cognitive_analyzer.analyze('¿Por qué un for es mejor que un while aquí?')
        assert analysis.bloom_level >= 3
    
    def test_engagement_score(self, cognitive_analyzer):
        """Verifica cálculo de engagement."""
        # Pregunta corta
        short = cognitive_analyzer.analyze('¿Qué es x?')
        
        # Pregunta elaborada
        long = cognitive_analyzer.analyze(
            'Estoy tratando de entender cómo funcionan los decoradores '
            'en Python y me pregunto si podrías explicarme la diferencia '
            'entre @staticmethod y @classmethod con ejemplos prácticos.'
        )
        
        assert long.engagement_score >= short.engagement_score


# ═══════════════════════════════════════════════════════════════════════
# TESTS: CONFIG
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Tests de configuración."""
    
    def test_presets(self):
        """Verifica presets de configuración."""
        from config import PRESETS, get_preset
        
        assert 'examen' in PRESETS
        assert 'repaso' in PRESETS
        
        examen = get_preset('examen')
        assert examen.max_daily_prompts == 3
        assert examen.block_direct_solutions is True
    
    def test_system_config(self):
        """Verifica configuración del sistema."""
        from config import SystemConfig
        
        config = SystemConfig()
        assert config.api_port == 8000
        assert config.database_url.endswith('.db')
    
    def test_llm_config(self):
        """Verifica configuración LLM."""
        from config import LLMConfig, LLMProvider
        
        config = LLMConfig.from_env()
        # Sin API key, debe ser mock
        assert config.provider == LLMProvider.MOCK


# ═══════════════════════════════════════════════════════════════════════
# TESTS: ANALYTICS MODULES
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyticsModules:
    """Tests de módulos de analytics."""
    
    def test_trust_dynamics(self):
        """Verifica TrustDynamicsAnalyzer."""
        from trust_dynamics import TrustDynamicsAnalyzer
        
        analyzer = TrustDynamicsAnalyzer()
        assert analyzer is not None
    
    def test_nd_patterns(self):
        """Verifica NeurodivergentPatternDetector."""
        from nd_patterns import NeurodivergentPatternDetector, InteractionEvent
        
        detector = NeurodivergentPatternDetector()
        
        # Crear eventos de prueba
        events = [
            InteractionEvent(
                event_id=f'e{i}',
                student_id='s1',
                timestamp=f'2026-02-22T10:{i:02d}:00',
                prompt='Test',
                detected_topics=['topic1'],
                bloom_level=2,
            )
            for i in range(5)
        ]
        
        for event in events:
            detector.process_event(event)
        
        # Debe poder obtener patrones (aunque estén vacíos)
        patterns = detector.get_patterns('s1')
        assert isinstance(patterns, list)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
