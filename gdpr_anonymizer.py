"""
GDPR ANONYMIZER — Capa de Anonimización y Consentimiento
═══════════════════════════════════════════════════════════════════════
Obligatorio antes de cualquier piloto con estudiantes reales.
El comité de ética de la UVa lo pedirá explícitamente.

PRINCIPIOS:
1. Pseudonimización: los módulos analíticos reciben IDs pseudónimos,
   la tabla de mapeo vive separada con acceso restringido.
2. Consentimiento granular: el estudiante elige qué datos permite.
3. Data minimization: solo se recoge lo necesario para la investigación.
4. Right to be forgotten: borrado completo de un participante.
5. Portabilidad: exportar datos de un participante en JSON.

CATEGORÍAS DE DATOS (consentimiento individual por categoría):
    interaction_logs    → prompts y respuestas (el más sensible)
    cognitive_profile   → perfiles de Bloom y engagement
    autonomy_tracking   → trayectorias de autonomía epistémica
    nd_patterns         → patrones neurodivergentes (máxima sensibilidad)
    behavioral_signals  → gaming, copy-paste, silencios

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set


DATA_CATEGORIES = {
    "interaction_logs": {
        "description": "Registro de preguntas al chatbot y respuestas recibidas",
        "sensitivity": "high",
        "required_for": ["O3_analytics"],
    },
    "cognitive_profile": {
        "description": "Perfil de nivel cognitivo (Bloom) y engagement",
        "sensitivity": "medium",
        "required_for": ["O3_analytics", "O2_learning_design"],
    },
    "autonomy_tracking": {
        "description": "Evolución de la autonomía de aprendizaje",
        "sensitivity": "medium",
        "required_for": ["O3_analytics"],
    },
    "nd_patterns": {
        "description": "Detección de patrones neurodivergentes funcionales",
        "sensitivity": "critical",
        "required_for": ["O2_udl_adaptation"],
    },
    "behavioral_signals": {
        "description": "Señales de comportamiento (gaming, copy-paste, silencios)",
        "sensitivity": "medium",
        "required_for": ["O3_analytics"],
    },
}


@dataclass
class ConsentRecord:
    """Registro de consentimiento de un participante."""
    participant_id: str            # ID real
    participant_type: str          # student | teacher
    pseudonym: str                 # ID pseudonimizado
    consent_given: bool = False
    consent_date: str = ""
    consent_version: str = "1.0"
    categories_consented: Set[str] = field(default_factory=set)
    withdrawal_date: str = ""
    is_active: bool = True


class GDPRAnonymizer:
    """
    Capa de anonimización para el ecosistema GENIE Learn.
    
    Se inserta ENTRE el middleware y los módulos analíticos:
    
    middleware.log_interaction(real_id) 
        → anonymizer.pseudonymize(real_id) 
            → analytics.analyze(pseudo_id)
    
    La tabla de mapeo real↔pseudónimo se almacena cifrada
    y separada de los datos analíticos.
    """

    def __init__(self, secret_key: Optional[str] = None):
        # Clave para HMAC — en producción, desde variable de entorno
        self._secret = (secret_key or os.getenv(
            "GENIE_ANONYMIZER_KEY", "genie-learn-dev-key-change-in-production"
        )).encode()

        # Mapeo real → pseudónimo (en producción: tabla cifrada en BD separada)
        self._id_map: Dict[str, str] = {}
        self._reverse_map: Dict[str, str] = {}

        # Registros de consentimiento
        self._consents: Dict[str, ConsentRecord] = {}

        # Datos que han sido borrados (right to be forgotten)
        self._forgotten: Set[str] = set()

    # ── PSEUDONIMIZACIÓN ──────────────────────────────────────

    def pseudonymize(self, real_id: str) -> str:
        """
        Genera pseudónimo determinista para un ID real.
        El mismo real_id siempre produce el mismo pseudónimo.
        Usa HMAC-SHA256 para que no sea reversible sin la clave.
        """
        if real_id in self._forgotten:
            raise ValueError(f"Participant {real_id} exercised right to be forgotten")

        if real_id in self._id_map:
            return self._id_map[real_id]

        # HMAC-SHA256: determinista + no reversible sin clave
        pseudo = "P_" + hmac.new(
            self._secret, real_id.encode(), hashlib.sha256
        ).hexdigest()[:12]

        self._id_map[real_id] = pseudo
        self._reverse_map[pseudo] = real_id
        return pseudo

    def depseudonymize(self, pseudo_id: str) -> Optional[str]:
        """
        Recupera ID real desde pseudónimo.
        SOLO accesible para investigadores autorizados con la clave.
        En producción: requiere 2FA + log de acceso.
        """
        return self._reverse_map.get(pseudo_id)

    def anonymize_interaction(self, interaction: dict) -> dict:
        """
        Anonimiza una interacción completa antes de pasarla a analytics.
        Elimina campos sensibles según el consentimiento del participante.
        """
        real_id = interaction.get("student_id", "")
        if real_id in self._forgotten:
            return {}

        consent = self._consents.get(real_id)
        pseudo = self.pseudonymize(real_id)

        # Copiar con ID pseudonimizado
        anonymized = {**interaction, "student_id": pseudo}

        # Eliminar campos según consentimiento
        if consent and consent.is_active:
            if "interaction_logs" not in consent.categories_consented:
                anonymized.pop("prompt_raw", None)
                anonymized.pop("response_delivered", None)
                anonymized["prompt_redacted"] = True

            if "nd_patterns" not in consent.categories_consented:
                anonymized.pop("functional_patterns", None)
                anonymized.pop("nd_signals", None)

            if "behavioral_signals" not in consent.categories_consented:
                anonymized.pop("copy_paste_score", None)
                anonymized.pop("gaming_suspicion", None)
        else:
            # Sin consentimiento: solo datos mínimos agregados
            return {
                "student_id": pseudo,
                "timestamp": interaction.get("timestamp", ""),
                "bloom_estimate": interaction.get("bloom_estimate", 0),
                "consent_status": "not_given",
            }

        # Siempre eliminar campos de identificación indirecta
        anonymized.pop("ip_address", None)
        anonymized.pop("user_agent", None)
        anonymized.pop("email", None)
        anonymized.pop("name", None)

        return anonymized

    # ── CONSENTIMIENTO ────────────────────────────────────────

    def register_consent(
        self,
        participant_id: str,
        participant_type: str,
        categories: List[str],
        version: str = "1.0",
    ) -> ConsentRecord:
        """Registra consentimiento informado de un participante."""
        pseudo = self.pseudonymize(participant_id)

        valid_categories = set(categories) & set(DATA_CATEGORIES.keys())

        record = ConsentRecord(
            participant_id=participant_id,
            participant_type=participant_type,
            pseudonym=pseudo,
            consent_given=True,
            consent_date=datetime.now().isoformat(),
            consent_version=version,
            categories_consented=valid_categories,
            is_active=True,
        )
        self._consents[participant_id] = record
        return record

    def withdraw_consent(self, participant_id: str) -> dict:
        """Retira consentimiento — datos futuros no se recogen."""
        if participant_id in self._consents:
            record = self._consents[participant_id]
            record.is_active = False
            record.withdrawal_date = datetime.now().isoformat()
            return {"status": "withdrawn", "participant": participant_id}
        return {"status": "not_found"}

    def has_consent(self, participant_id: str, category: str) -> bool:
        """Verifica si un participante tiene consentimiento para una categoría."""
        consent = self._consents.get(participant_id)
        if not consent or not consent.is_active:
            return False
        return category in consent.categories_consented

    # ── RIGHT TO BE FORGOTTEN ─────────────────────────────────

    def forget_participant(self, participant_id: str) -> dict:
        """
        Borrado completo — Artículo 17 GDPR.
        Elimina mapeo, consentimiento, y marca para exclusión futura.
        Los módulos analíticos deben verificar este estado.
        """
        pseudo = self._id_map.get(participant_id, "")

        # Borrar mapeo
        self._id_map.pop(participant_id, None)
        if pseudo:
            self._reverse_map.pop(pseudo, None)

        # Borrar consentimiento
        self._consents.pop(participant_id, None)

        # Marcar como olvidado
        self._forgotten.add(participant_id)

        return {
            "status": "forgotten",
            "participant_id": participant_id,
            "pseudonym_deleted": pseudo,
            "timestamp": datetime.now().isoformat(),
            "note": "All analytical modules must purge data for this participant",
        }

    # ── DATA PORTABILITY ──────────────────────────────────────

    def export_participant_data(self, participant_id: str) -> dict:
        """
        Exporta todos los datos de un participante — Artículo 20 GDPR.
        En producción: consulta todas las tablas de persistence_layer.
        """
        consent = self._consents.get(participant_id)
        return {
            "participant_id": participant_id,
            "pseudonym": self._id_map.get(participant_id, "unknown"),
            "consent": {
                "given": consent.consent_given if consent else False,
                "date": consent.consent_date if consent else "",
                "categories": list(consent.categories_consented) if consent else [],
                "active": consent.is_active if consent else False,
            },
            "export_date": datetime.now().isoformat(),
            "note": "In production, this includes all data from persistence_layer tables",
        }

    # ── REPORTS ───────────────────────────────────────────────

    def get_consent_summary(self) -> dict:
        """Resumen de consentimientos para el comité de ética."""
        active = [c for c in self._consents.values() if c.is_active]
        withdrawn = [c for c in self._consents.values() if not c.is_active]

        category_coverage = {}
        for cat in DATA_CATEGORIES:
            consented = sum(1 for c in active if cat in c.categories_consented)
            category_coverage[cat] = {
                "consented": consented,
                "total_active": len(active),
                "coverage_pct": round(consented / max(len(active), 1) * 100, 1),
            }

        return {
            "total_participants": len(self._consents),
            "active_consents": len(active),
            "withdrawn_consents": len(withdrawn),
            "forgotten_participants": len(self._forgotten),
            "category_coverage": category_coverage,
            "consent_version": "1.0",
        }


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    anon = GDPRAnonymizer()

    # Registrar consentimientos
    c1 = anon.register_consent("est_real_001", "student",
        ["interaction_logs", "cognitive_profile", "autonomy_tracking"])
    c2 = anon.register_consent("est_real_002", "student",
        ["cognitive_profile"])  # No consiente interaction_logs
    c3 = anon.register_consent("prof_real_001", "teacher",
        ["interaction_logs", "cognitive_profile", "behavioral_signals"])

    print(f"Estudiante 1 pseudónimo: {c1.pseudonym}")
    print(f"Estudiante 2 pseudónimo: {c2.pseudonym}")

    # Anonimizar interacción
    interaction = {
        "student_id": "est_real_001",
        "prompt_raw": "¿Cómo funciona un bucle for?",
        "bloom_estimate": 2,
        "copy_paste_score": 0.1,
        "response_delivered": "Un bucle for tiene tres partes...",
    }
    anonymized = anon.anonymize_interaction(interaction)
    print(f"\nInteracción anonimizada (con consentimiento completo):")
    print(f"  student_id: {anonymized['student_id']}")
    print(f"  prompt visible: {'prompt_raw' in anonymized}")

    # Estudiante 2 sin consentimiento de logs
    interaction2 = {**interaction, "student_id": "est_real_002"}
    anonymized2 = anon.anonymize_interaction(interaction2)
    print(f"\nInteracción anonimizada (sin consentimiento logs):")
    print(f"  prompt visible: {'prompt_raw' in anonymized2}")
    print(f"  redacted: {anonymized2.get('prompt_redacted', False)}")

    # Right to be forgotten
    result = anon.forget_participant("est_real_002")
    print(f"\nDerecho al olvido: {result['status']}")

    # Resumen
    summary = anon.get_consent_summary()
    print(f"\nResumen consentimientos:")
    print(f"  Activos: {summary['active_consents']}")
    print(f"  Olvidados: {summary['forgotten_participants']}")

    print("\n✓ GDPR Anonymizer operativo")
