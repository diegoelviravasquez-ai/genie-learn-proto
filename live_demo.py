"""
LIVE DEMO — Simulación de Sesión en Vivo
═══════════════════════════════════════════════════════════════
PROBLEMA QUE ATACA:
Permitir al docente ver en tiempo real cómo un estudiante ficticio
interactúa con el chatbot, con momentos de frustración y recuperación,
para evaluar el efecto de cambios de configuración.

FUNDAMENTACIÓN TEÓRICA:
- Think-aloud protocols (Ericsson & Simon, 1980)
- Simulación de estudiantes para formación docente (Girod et al., 2003)

POSICIÓN EN EL ECOSISTEMA:
live_demo → app.py (botón Demo en vivo)
"""

import random
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator
from datetime import datetime

logger = logging.getLogger(__name__)

# Preguntas realistas por tema (Bloom ascendente)
STUDENT_PROMPTS = {
    "variables": [
        "¿Qué es una variable?",
        "¿Cómo declaro una variable en Java?",
        "¿Cuál es la diferencia entre int y double?",
        "¿Puedo cambiar el valor de una variable después de declararla?",
    ],
    "bucles": [
        "¿Qué es un bucle?",
        "¿Cuál es la diferencia entre for y while?",
        "¿Cómo recorro un array con un for?",
        "¿Qué pasa si no pongo condición de parada en un while?",
    ],
    "funciones": [
        "¿Para qué sirven las funciones?",
        "¿Cómo paso parámetros a una función?",
        "¿Qué es el return?",
        "¿Puedo llamar a una función desde otra función?",
    ],
    "recursión": [
        "¿Qué es la recursión?",
        "¿Qué es el caso base?",
        "¿Por qué mi recursión no termina nunca?",
        "¿Cuándo es mejor recursión que un bucle?",
    ],
}

FRUSTRATION_PROMPTS = [
    "no entiendo",
    "da igual, no lo pillo",
    "esto no me sale",
    "¿me lo resuelves tú?",
    "no funciona",
]

RECOVERY_PROMPTS = [
    "ah vale, creo que ya lo vi",
    "gracias, voy a intentarlo",
    "ok, lo intento de nuevo",
]


@dataclass
class SimulatedMessage:
    """Mensaje simulado del estudiante."""
    prompt: str
    topic: str
    bloom_trend: int  # -1, 0, 1
    is_frustration: bool
    is_recovery: bool
    timestamp: str


def simulate_session(
    orchestrator,
    student_id: str = "demo_estudiante",
    interval_seconds: float = 5.0,
    num_messages: int = 10,
    frustration_prob: float = 0.2,
) -> Generator[Dict, None, None]:
    """
    Genera interacciones realistas de un estudiante ficticio.
    
    El estudiante "aprende" (Bloom sube gradualmente) con momentos de
    frustración y recuperación.
    
    Uso:
        for event in simulate_session(orch, interval_seconds=5, num_messages=8):
            # event = {"type": "message"|"frustration"|"recovery", "result": ..., "prompt": ...}
            yield event
    """
    topics = list(STUDENT_PROMPTS.keys())
    current_bloom = 1
    topic_index = 0
    frustration_count = 0

    for i in range(num_messages):
        # Decidir si frustración o recuperación
        if random.random() < frustration_prob and frustration_count < 2:
            prompt = random.choice(FRUSTRATION_PROMPTS)
            topic = random.choice(topics)
            is_frustration = True
            is_recovery = False
            bloom_trend = -1
            frustration_count += 1
        elif frustration_count > 0 and random.random() < 0.5:
            prompt = random.choice(RECOVERY_PROMPTS)
            topic = random.choice(topics)
            is_frustration = False
            is_recovery = True
            bloom_trend = 1
            frustration_count = max(0, frustration_count - 1)
        else:
            topic = topics[topic_index % len(topics)]
            prompts_list = STUDENT_PROMPTS[topic]
            level = min(current_bloom, len(prompts_list) - 1)
            prompt = prompts_list[level]
            is_frustration = False
            is_recovery = False
            bloom_trend = 1 if random.random() < 0.6 else 0
            current_bloom = min(6, current_bloom + (1 if bloom_trend > 0 else 0))
            topic_index += 1

        result = orchestrator.process_interaction(student_id, prompt)

        event = {
            "type": "frustration" if is_frustration else ("recovery" if is_recovery else "message"),
            "prompt": prompt,
            "result": result,
            "topic": topic,
            "bloom_trend": bloom_trend,
            "is_frustration": is_frustration,
            "is_recovery": is_recovery,
            "timestamp": datetime.now().isoformat(),
        }

        yield event
        time.sleep(interval_seconds)
