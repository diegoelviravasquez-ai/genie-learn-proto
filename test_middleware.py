"""
Tests del middleware pedagógico.

Estos tests no verifican que "el código no se rompe" — verifican que
las DECISIONES PEDAGÓGICAS se implementan correctamente. Cada test
corresponde a un escenario real que un docente esperaría.

Convención: test_[quién]_[hace_qué]_[espera_qué]
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware import PedagogicalMiddleware, PedagogicalConfig


@pytest.fixture
def strict_config():
    """Config de un docente estricto: socrático, sin soluciones, con límites."""
    return PedagogicalConfig(
        max_daily_prompts=5,
        scaffolding_mode="socratic",
        block_direct_solutions=True,
        forced_hallucination_pct=0.0,
        use_rag=True,
        no_context_behavior="refuse",
    )

@pytest.fixture
def permissive_config():
    """Config de un docente permisivo: respuesta directa, sin límites estrictos."""
    return PedagogicalConfig(
        max_daily_prompts=50,
        scaffolding_mode="direct",
        block_direct_solutions=False,
        use_rag=True,
        no_context_behavior="general",
    )

@pytest.fixture
def strict_middleware(strict_config):
    return PedagogicalMiddleware(strict_config)

@pytest.fixture
def permissive_middleware(permissive_config):
    return PedagogicalMiddleware(permissive_config)


# ──────────────────────────────────────────────
# LÍMITES DE USO
# ──────────────────────────────────────────────

class TestDailyLimits:
    """
    El límite diario de prompts es una 'desirable difficulty' (Bjork, 1994):
    fuerza al estudiante a pensar antes de preguntar.
    """

    def test_student_dentro_del_limite_puede_preguntar(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "¿qué es un int?")
        assert result["allowed"] is True

    def test_student_que_agota_limite_es_bloqueado(self, strict_middleware):
        """5 preguntas = límite. La 6ª debe ser rechazada."""
        for i in range(5):
            strict_middleware.pre_process("est_01", f"pregunta {i}")
        result = strict_middleware.pre_process("est_01", "una más")
        assert result["allowed"] is False
        assert "límite" in result["block_reason"].lower()

    def test_limite_es_por_estudiante_no_global(self, strict_middleware):
        """El límite de est_01 no afecta a est_02."""
        for i in range(5):
            strict_middleware.pre_process("est_01", f"pregunta {i}")
        result = strict_middleware.pre_process("est_02", "mi primera pregunta")
        assert result["allowed"] is True

    def test_bloqueo_incluye_explicacion_pedagogica(self, strict_middleware):
        """El mensaje de bloqueo no es un error técnico — es una explicación pedagógica."""
        for i in range(5):
            strict_middleware.pre_process("est_01", f"q{i}")
        result = strict_middleware.pre_process("est_01", "más")
        assert "reflexión" in result["block_reason"].lower() or "pedagógic" in result["block_reason"].lower()


# ──────────────────────────────────────────────
# SCAFFOLDING SOCRÁTICO
# ──────────────────────────────────────────────

class TestSocraticScaffolding:
    """
    El scaffolding escala: primero preguntas, luego pistas, luego ejemplos,
    luego explicación. Esto es Wood, Bruner & Ross (1976).
    """

    def test_primera_pregunta_es_nivel_socratico(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "¿qué es un bucle?")
        assert result["scaffolding_level"] == 0
        assert "SOCRÁTICO" in result["system_prompt"]

    def test_scaffolding_escala_con_intentos(self, strict_middleware):
        """Después de 2 intentos en el mismo tema, sube de nivel."""
        strict_middleware.pre_process("est_01", "¿qué es un bucle?")
        # El post_process es quien escala el nivel
        strict_middleware.post_process("est_01", "respuesta simulada")
        strict_middleware.pre_process("est_01", "sigo sin entender bucles")
        strict_middleware.post_process("est_01", "respuesta simulada")
        
        # Tercer intento: debería haber escalado
        result = strict_middleware.pre_process("est_01", "bucles otra vez")
        assert result["scaffolding_level"] >= 1

    def test_reset_vuelve_a_nivel_cero(self, strict_middleware):
        """El docente puede resetear el scaffolding de un estudiante."""
        strict_middleware.pre_process("est_01", "pregunta")
        strict_middleware.post_process("est_01", "resp")
        strict_middleware.post_process("est_01", "resp")
        strict_middleware.reset_student("est_01")
        result = strict_middleware.pre_process("est_01", "nueva pregunta")
        assert result["scaffolding_level"] == 0


# ──────────────────────────────────────────────
# DETECCIÓN DE COPY-PASTE
# ──────────────────────────────────────────────

class TestCopyPasteDetection:
    """
    La detección de copy-paste no es un sistema antiplagio — es una señal
    para el docente de que el estudiante posiblemente está copiando enunciados
    en vez de formular preguntas propias (señal de bajo engagement).
    """

    def test_pregunta_corta_no_dispara_alerta(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "¿qué es un array?")
        assert result["copy_paste_score"] < 0.5

    def test_enunciado_largo_sin_interrogacion_dispara_alerta(self, strict_middleware):
        enunciado = (
            "Ejercicio 3. Dado el siguiente array de enteros, implementar un método "
            "que reciba como parámetro el array y devuelva la suma de todos los elementos "
            "que sean pares. Utilizar un bucle for para recorrer el array. No se permite "
            "el uso de métodos de la clase Arrays."
        )
        result = strict_middleware.pre_process("est_01", enunciado)
        assert result["copy_paste_score"] > 0.5

    def test_copypaste_no_bloquea_solo_registra(self, strict_middleware):
        """Copy-paste genera alerta pero NO bloquea — el docente decide qué hacer."""
        enunciado = "Ejercicio 5. Se pide implementar un programa que resuelva..."
        result = strict_middleware.pre_process("est_01", enunciado)
        assert result["allowed"] is True  # No bloquea
        assert result["copy_paste_score"] > 0   # Pero registra


# ──────────────────────────────────────────────
# DETECCIÓN DE TOPICS
# ──────────────────────────────────────────────

class TestTopicDetection:
    """Los topics se detectan para que el docente sepa qué temas generan más dudas."""

    def test_pregunta_sobre_bucles_detecta_topic(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "¿cómo funciona el bucle for?")
        assert "bucles" in result["detected_topics"]

    def test_pregunta_sobre_arrays_detecta_topic(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "no entiendo los arrays en Java")
        assert "arrays" in result["detected_topics"]

    def test_pregunta_ambigua_detecta_otro(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "estoy perdido con todo")
        assert "otro" in result["detected_topics"]

    def test_pregunta_multitema_detecta_varios(self, strict_middleware):
        result = strict_middleware.pre_process(
            "est_01",
            "¿cómo uso un bucle for para recorrer un array?"
        )
        assert "bucles" in result["detected_topics"]
        assert "arrays" in result["detected_topics"]


# ──────────────────────────────────────────────
# SYSTEM PROMPT DINÁMICO
# ──────────────────────────────────────────────

class TestSystemPrompt:
    """
    El system prompt se construye dinámicamente según las configuraciones
    del docente. Esto es la implementación del middleware del paper LAK 2026.
    """

    def test_modo_socratico_genera_instrucciones_socraticas(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "pregunta")
        assert "SOCRÁTICO" in result["system_prompt"] or "preguntas" in result["system_prompt"].lower()

    def test_modo_directo_no_restringe(self, permissive_middleware):
        result = permissive_middleware.pre_process("est_01", "pregunta")
        # En modo directo, no debería haber instrucciones de restricción socrática
        assert "SOCRÁTICO" not in result["system_prompt"]

    def test_bloqueo_soluciones_aparece_en_prompt(self, strict_middleware):
        result = strict_middleware.pre_process("est_01", "resuélveme esto")
        assert "solución" in result["system_prompt"].lower() or "NO" in result["system_prompt"]

    def test_role_play_se_inyecta(self):
        config = PedagogicalConfig(
            role_play="Eres un profesor de Hogwarts que enseña programación con metáforas mágicas"
        )
        mw = PedagogicalMiddleware(config)
        result = mw.pre_process("est_01", "hola")
        assert "Hogwarts" in result["system_prompt"]


# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

class TestLogging:
    """Todo lo que pasa se registra. Sin datos, no hay investigación."""

    def test_interaccion_se_registra(self, strict_middleware):
        pre = strict_middleware.pre_process("est_01", "pregunta test")
        post = strict_middleware.post_process("est_01", "respuesta test")
        strict_middleware.log_interaction(
            student_id="est_01",
            prompt_raw="pregunta test",
            pre_result=pre,
            response_raw="respuesta test",
            post_result=post,
            response_time_ms=500,
        )
        assert len(strict_middleware.interaction_logs) == 1

    def test_analytics_summary_con_datos(self, strict_middleware):
        pre = strict_middleware.pre_process("est_01", "¿qué es un for?")
        post = strict_middleware.post_process("est_01", "respuesta")
        strict_middleware.log_interaction("est_01", "q", pre, "r", post, 300)
        summary = strict_middleware.get_analytics_summary()
        assert summary["total_interactions"] == 1
        assert summary["unique_students"] == 1
