"""
DINÁMICAS DE CONFIANZA ESTUDIANTE-IA
======================================
Módulo de detección de patrones de confianza/desconfianza en la
relación estudiante-chatbot. Conecta con el trabajo previo de
IA Trust Nexus (evaluación de confianza bajo EU AI Act).

Problema de investigación:
  ¿Cómo evoluciona la confianza del estudiante en el chatbot
  a lo largo de las interacciones? ¿Qué señales conductuales
  predicen sobre-dependencia (automation bias) o infra-uso
  (algorithm aversion)?

Marco teórico:
  - Lee & See (2004). Trust in automation: 3 dimensiones
    (performance, process, purpose). Aplicado aquí al contexto
    educativo: el estudiante confía si el chatbot (a) responde bien,
    (b) lo hace de forma comprensible, (c) para un fin que percibe legítimo.
  - Parasuraman & Riley (1997). Misuse (sobre-confianza) vs. Disuse
    (infra-confianza) de la automatización.
  - EU AI Act (2024), Art. 14: "Human oversight" — el derecho del
    usuario a NO seguir la recomendación de la IA.
  - Kaur et al. (2022). Sensibility of automation trust in AI
    educational systems — la confianza en IA educativa es diferente
    a la confianza en IA genérica porque el OBJETIVO es aprender,
    no obtener la respuesta correcta.

La paradoja central:
  Un estudiante que confía DEMASIADO en el chatbot no aprende
  (copia respuestas). Un estudiante que confía DEMASIADO POCO
  tampoco (no lo usa). El sweet spot es la CONFIANZA CALIBRADA:
  confiar lo suficiente para usarlo como herramienta de
  aprendizaje, pero no tanto como para delegar el pensamiento.

Operacionalización:
  No medimos confianza con encuestas (eso es post-hoc). Medimos
  SEÑALES CONDUCTUALES en tiempo real:
  - Latencia entre respuesta del chatbot y siguiente prompt
    → poca latencia = no lee la respuesta (sobre-confianza o frustración)
  - Reformulaciones después de una respuesta
    → indica que el estudiante procesa y rebota
  - Copy-paste de respuestas del chatbot
    → sobre-confianza en el contenido
  - Preguntas de verificación ("¿estás seguro?", "¿esto es correcto?")
    → señal de confianza calibrada (deseable)
  - Abandono de sesión después de respuesta socrática
    → posible frustración con el scaffolding (infra-confianza en el método)

Autor: Diego Elvira Vásquez
Conexión: IA Trust Nexus (motores Riesgo/Confianza) adaptado a contexto educativo.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import re


@dataclass
class TrustSignal:
    """Una señal conductual de confianza/desconfianza."""
    timestamp: str
    student_id: str
    signal_type: str        # "verification", "reformulation", "quick_follow", "abandonment", "acceptance"
    trust_direction: float  # -1.0 (desconfianza) a +1.0 (sobre-confianza). 0 = calibrada
    description: str
    raw_data: dict


class TrustDynamicsAnalyzer:
    """
    Analiza patrones de confianza en la interacción estudiante-chatbot.
    
    Output: un score de CALIBRACIÓN de confianza por estudiante.
    - Score cercano a 0.0 = confianza calibrada (óptimo)
    - Score > 0.5 = sobre-confianza (riesgo: automation bias)
    - Score < -0.5 = infra-confianza (riesgo: desuso)
    """

    # Marcadores de verificación (señal POSITIVA de confianza calibrada)
    VERIFICATION_MARKERS = [
        "estás seguro", "es correcto", "eso es cierto",
        "puedo fiarme", "está bien esto", "confirma",
        "seguro que", "no me estarás", "verificar",
        "contrastar", "segundo opinión", "otra fuente",
    ]

    # Marcadores de reformulación (señal POSITIVA de procesamiento activo)
    REFORMULATION_MARKERS = [
        "entonces lo que dices", "es decir", "o sea que",
        "a ver si entiendo", "si he entendido bien",
        "reformulo", "para asegurarme", "corrígeme",
        "lo que quieres decir es", "en otras palabras",
    ]

    # Marcadores de aceptación acrítica (señal de SOBRE-CONFIANZA)
    UNCRITICAL_ACCEPTANCE = [
        "ok gracias", "perfecto", "vale genial",
        "ya entendí gracias", "listo", "ok",
        "siguiente pregunta", "ahora dime",
    ]

    # Marcadores de frustración / rechazo (señal de INFRA-CONFIANZA)
    FRUSTRATION_MARKERS = [
        "no me sirve", "eso no es lo que pregunté",
        "dame la respuesta directa", "no quiero preguntas",
        "deja de preguntar", "responde ya", "no me ayudas",
        "voy a buscar en google", "mejor chatgpt",
        "no entiendes", "inútil",
    ]

    def __init__(self):
        self.signals: list[TrustSignal] = []
        self.interaction_timestamps: dict[str, list[datetime]] = {}

    def analyze_prompt(self, student_id: str, prompt: str, previous_response: str = "") -> TrustSignal:
        """Analiza un prompt buscando señales de confianza."""
        prompt_lower = prompt.lower().strip()

        # Verificación
        if any(m in prompt_lower for m in self.VERIFICATION_MARKERS):
            signal = TrustSignal(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                signal_type="verification",
                trust_direction=0.0,  # Calibrada — esto es lo deseable
                description="Verificación activa: el estudiante cuestiona la respuesta antes de aceptarla.",
                raw_data={"matched": [m for m in self.VERIFICATION_MARKERS if m in prompt_lower]},
            )
        # Reformulación
        elif any(m in prompt_lower for m in self.REFORMULATION_MARKERS):
            signal = TrustSignal(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                signal_type="reformulation",
                trust_direction=0.1,  # Ligeramente positiva (procesa activamente)
                description="Reformulación: el estudiante re-expresa la respuesta del chatbot en sus propias palabras.",
                raw_data={"matched": [m for m in self.REFORMULATION_MARKERS if m in prompt_lower]},
            )
        # Aceptación acrítica
        elif any(m in prompt_lower for m in self.UNCRITICAL_ACCEPTANCE) and len(prompt.split()) < 8:
            signal = TrustSignal(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                signal_type="uncritical_acceptance",
                trust_direction=0.6,  # Sobre-confianza
                description="Aceptación acrítica: respuesta breve sin procesamiento visible.",
                raw_data={"matched": [m for m in self.UNCRITICAL_ACCEPTANCE if m in prompt_lower]},
            )
        # Frustración
        elif any(m in prompt_lower for m in self.FRUSTRATION_MARKERS):
            signal = TrustSignal(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                signal_type="frustration",
                trust_direction=-0.7,  # Infra-confianza
                description="Frustración: posible rechazo del scaffolding o del sistema.",
                raw_data={"matched": [m for m in self.FRUSTRATION_MARKERS if m in prompt_lower]},
            )
        else:
            signal = TrustSignal(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                signal_type="neutral",
                trust_direction=0.0,
                description="Sin señal clara de confianza/desconfianza.",
                raw_data={},
            )

        self.signals.append(signal)
        return signal

    def analyze_timing(self, student_id: str, current_timestamp: datetime) -> Optional[TrustSignal]:
        """
        Analiza el timing entre interacciones.
        Respuesta inmediata (<5s) después de una respuesta larga = no leyó.
        """
        if student_id not in self.interaction_timestamps:
            self.interaction_timestamps[student_id] = []

        timestamps = self.interaction_timestamps[student_id]
        self.interaction_timestamps[student_id].append(current_timestamp)

        if len(timestamps) < 1:
            return None

        last_time = timestamps[-1]
        delta = (current_timestamp - last_time).total_seconds()

        if delta < 5:
            return TrustSignal(
                timestamp=current_timestamp.isoformat(),
                student_id=student_id,
                signal_type="quick_follow",
                trust_direction=0.4,  # Posible sobre-confianza
                description=f"Respuesta muy rápida ({delta:.0f}s) — posible falta de lectura de la respuesta anterior.",
                raw_data={"delta_seconds": delta},
            )
        return None

    def get_student_trust_profile(self, student_id: str) -> dict:
        """
        Calcula el perfil de confianza de un estudiante.
        
        Returns:
            calibration_score: -1.0 (infra) a +1.0 (sobre). 0 = calibrada
            trust_category: "calibrated", "over_reliant", "under_reliant", "unknown"
            signal_distribution: conteo por tipo de señal
            recommendation: consejo para el docente
        """
        student_signals = [s for s in self.signals if s.student_id == student_id]

        if len(student_signals) < 3:
            return {
                "student_id": student_id,
                "calibration_score": 0.0,
                "trust_category": "unknown",
                "n_signals": len(student_signals),
                "recommendation": "Datos insuficientes. Se necesitan más interacciones.",
            }

        # Media ponderada de trust_direction (señales recientes pesan más)
        n = len(student_signals)
        weighted_sum = 0
        weight_total = 0
        for i, signal in enumerate(student_signals):
            weight = 1 + (i / n)  # más reciente = más peso
            weighted_sum += signal.trust_direction * weight
            weight_total += weight

        calibration = weighted_sum / weight_total if weight_total > 0 else 0

        # Distribución de señales
        distribution = {}
        for s in student_signals:
            distribution[s.signal_type] = distribution.get(s.signal_type, 0) + 1

        # Categorización
        if abs(calibration) < 0.2:
            category = "calibrated"
            recommendation = (
                "Confianza calibrada. El estudiante usa el chatbot como herramienta "
                "de aprendizaje, no como oráculo. Mantener configuración actual."
            )
        elif calibration > 0.2:
            category = "over_reliant"
            recommendation = (
                "Sobre-dependencia detectada. Considerar: (1) aumentar scaffolding socrático, "
                "(2) activar alucinaciones pedagógicas para forzar lectura crítica, "
                "(3) reducir límite de prompts diarios."
            )
        else:
            category = "under_reliant"
            recommendation = (
                "Infra-uso o frustración detectada. Considerar: (1) reducir scaffolding "
                "a modo directo temporalmente, (2) verificar que los materiales RAG son "
                "adecuados, (3) contactar al estudiante directamente."
            )

        return {
            "student_id": student_id,
            "calibration_score": round(calibration, 3),
            "trust_category": category,
            "n_signals": n,
            "signal_distribution": distribution,
            "recommendation": recommendation,
        }
