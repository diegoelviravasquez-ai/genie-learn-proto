"""
CAPA DE ANÁLISIS PRAGMÁTICO — Parche diferencial para cognitive_analyzer.py
═══════════════════════════════════════════════════════════════════════════════
PROBLEMA DOCUMENTADO:
El clasificador de Bloom en cognitive_analyzer.py usa keywords y regexes.
El test test_diferencia_entre_es_comprender_o_analizar captura el síntoma:
"¿por qué?" no siempre es análisis. El mismo marcador lingüístico ("por qué")
activa Bloom 4 (ANALIZAR) aunque la intención sea Bloom 3 (APLICAR) o incluso
Bloom 2 (COMPRENDER), dependiendo del contexto pragmático.

FUNDAMENTO TEÓRICO:
───────────────────
La distinción que los keywords no capturan es PRAGMÁTICA, no léxica. Reside
en la relación entre el enunciado y la situación que lo genera.

Austin (1962) y Searle (1969) distinguen:
  - Acto locucionario: lo que literalmente se dice ("¿por qué no funciona?")
  - Acto ilocucionario: la intención comunicativa (diagnosis de error → Bloom 3)
  - Acto perlocucionario: el efecto esperado (recibir orientación para corregir)

El clasificador actual captura el nivel locucionario. Esta capa añade el nivel
ilocucionario — lo que determina el nivel real de Bloom.

CUATRO HEURÍSTICAS:
───────────────────
H1. INDEXICAL vs SIMBÓLICO
    Goffman (1974), "Frame Analysis": el enunciado puede orientarse hacia
    una instancia concreta (indexical: "este código", "esta función", "el error
    de la línea 5") o hacia una relación abstracta (simbólico: "los bucles",
    "la recursión", "la diferencia entre").
    
    "¿por qué no me funciona este código?" → indexical → APLICAR (troubleshooting)
    "¿por qué los bucles for son más eficientes?" → simbólico → ANALIZAR

H2. FOCO EN SÍNTOMA vs FOCO EN MECANISMO
    Wineburg (1991), "On the Reading of Historical Texts": el experto busca
    el mecanismo causal; el novato busca el síntoma o la corrección inmediata.
    
    "¿por qué me sale este error?" → foco en síntoma → APLICAR
    "¿por qué el compilador no puede inferir el tipo?" → foco en mecanismo → ANALIZAR

H3. CORRECCIÓN BUSCADA vs COMPRENSIÓN BUSCADA
    Chi & Wylie (2014), ICAP: Constructive (propio output) vs Active (procesar).
    Si hay señal de que el estudiante quiere ARREGLAR algo concreto, la profundidad
    epistémica es menor aunque use "por qué".
    
    Señales de corrección: "no funciona", "da error", "falla", "se rompe",
    "no compila", primera persona + código ("mi función", "mi bucle").
    
    Señales de comprensión: "cuándo", "en qué caso", "cuál es la lógica",
    tercera persona + abstracción ("los punteros", "la recursión").

H4. PRESENCIA DE OBJETO ABSTRACTO vs CONCRETO
    Bernstein (1971), códigos lingüísticos: código elaborado (lengua
    independiente del contexto, abstracta) vs código restringido (dependiente
    del contexto inmediato, concreta).
    
    Objeto abstracto: "la complejidad algorítmica", "el polimorfismo", "la herencia"
    Objeto concreto: "mi for", "este while", "el código que tienes arriba", "esto"

INTEGRACIÓN:
La capa REFINA el output de CognitiveAnalyzer. Toma el nivel detectado y,
si detecta señal pragmática contradictoria, aplica un ajuste con penalización
de confianza. NO sobreescribe: modula.

Implementación sin NLP pesado: regex + heurísticas de reglas. False positive rate
aceptable para un prototipo de demo. En producción, reemplazar H1 y H4 con
classificación por embeddings.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import re
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# PATRONES PRAGMÁTICOS
# ═══════════════════════════════════════════════════════════════════════

# H1: Marcadores indexicales (señalan instancia concreta → baja abstracción)
INDEXICAL_MARKERS = [
    r"\b(este|esta|esto|estos|estas)\b",          # deícticos
    r"\b(el\s+código|la\s+función|el\s+bucle|el\s+programa)\s+que\b",
    r"\b(línea|linea)\s+\d+",                     # referencia a línea concreta
    r"\b(aquí|acá|arriba|abajo)\b",               # deícticos espaciales
    r"\bmi\s+(código|función|programa|solución|for|while|variable|clase)\b",
    r"\bel\s+(error|fallo|bug)\s+(que\s+me\s+sale|que\s+tengo|de\s+arriba)\b",
    r"\besto\b",                                  # pronombre deíctico de máxima proximidad
    r"\b(no\s+me\s+funciona|no\s+me\s+compila|no\s+me\s+corre)\b",
]

# H1: Marcadores simbólicos (señalan relación abstracta → alta abstracción)
SYMBOLIC_MARKERS = [
    r"\b(los|las|un|una|el|la)\s+(bucles?|arrays?|punteros?|recursiones?|funciones?|clases?|objetos?)\s+(en\s+general|normalmente|siempre)\b",
    r"\b(cuándo\s+se\s+usa|cuándo\s+usar|cuándo\s+es\s+mejor)\b",
    r"\b(la\s+diferencia\s+entre|diferencia\s+conceptual|qué\s+distingue)\b",
    r"\b(en\s+qué\s+(caso|contexto|situación))\b",
    r"\b(por\s+qué\s+existe|para\s+qué\s+existe|para\s+qué\s+sirve\s+la)\b",
    r"\b(ventajas?\s+y\s+desventajas?|pros?\s+y\s+contras?|trade.?off)\b",
    r"\b(la\s+lógica\s+detrás|el\s+mecanismo|cómo\s+funciona\s+internamente)\b",
    # Comparaciones de eficiencia abstractas (sin referencia a instancia concreta)
    r"\b(más\s+eficiente\s+(que|para)|mejor\s+que\s+.{0,20}(en|para|cuando))\b",
    r"\b(for\s+.{0,15}(más\s+eficiente|mejor|versus|vs)\s+.{0,15}while)\b",
    r"\b(while\s+.{0,15}(más\s+eficiente|mejor|versus|vs)\s+.{0,15}for)\b",
    r"\b(para\s+recorrer|para\s+iterar)\s+.{0,20}(siempre|cuando|mejor)\b",
]

# H2: Foco en síntoma (troubleshooting → APLICAR)
SYMPTOM_FOCUS_MARKERS = [
    r"no\s+me\s+(funciona|compila|corre|sale|ejecuta)",
    r"\b(no\s+funciona|no\s+compila|da\s+error|falla|se\s+rompe|no\s+corre)\b",
    r"\b(error\s+de\s+compilación|null\s+pointer|index\s+out\s+of|segmentation\s+fault)\b",
    r"\b(qué\s+(está|estoy)\s+(mal|haciendo\s+mal)|dónde\s+está\s+el\s+error)\b",
    r"\b(no\s+me\s+sale\s+el\s+resultado|me\s+da\s+un\s+número\s+raro)\b",
    r"\b(arrégl|corrij|soluciona)\b",
    r"tengo\s+un\s+(error|fallo|problema|bug)",
    r"me\s+(sale|da)\s+(un\s+)?(error|excepción|fallo)",
]

# H2: Foco en mecanismo (comprensión causal → ANALIZAR)
MECHANISM_FOCUS_MARKERS = [
    r"\b(internamente|bajo\s+el\s+capó|a\s+bajo\s+nivel)\b",
    r"\b(cómo\s+lo\s+procesa|cómo\s+lo\s+almacena|cómo\s+lo\s+gestiona)\b",
    r"\b(la\s+razón\s+por\s+la\s+que|el\s+motivo\s+por\s+el\s+que)\b",
    r"\b(qué\s+implica|qué\s+consecuencias|qué\s+impacto)\b",
    r"\b(por\s+qué\s+(es\s+mejor|es\s+preferible|se\s+recomienda))\b",
]

# H3: Señales de corrección buscada (reducir nivel)
CORRECTION_SEEKING_MARKERS = [
    r"\b(corrige|corrígeme|arregla|soluciona|arréglame)\b",
    r"\b(cómo\s+lo\s+arreglo|cómo\s+lo\s+corrijo|cómo\s+lo\s+soluciono)\b",
    r"\b(qué\s+cambio|qué\s+modifico|qué\s+pongo\s+en\s+vez\s+de)\b",
    r"\b(hay\s+algo\s+mal|algo\s+falla)\b",
]

# H4: Objetos abstractos (indicadores de código elaborado → alta abstracción)
ABSTRACT_OBJECT_MARKERS = [
    r"\b(complejidad\s+(algorítmica|temporal|espacial)|big.?o)\b",
    r"\b(polimorfismo|herencia|encapsulación|abstracción)\b",
    r"\b(tipo\s+de\s+dato\s+abstracto|estructura\s+de\s+datos)\b",
    r"\b(paradigma|programación\s+(orientada|funcional|imperativa))\b",
    r"\b(algoritmo\s+de|técnica\s+de|patrón\s+de\s+diseño)\b",
    r"\b(eficiencia|rendimiento\s+(computacional|espacial|temporal))\b",
]

# H4: Objetos concretos (código restringido → baja abstracción)
CONCRETE_OBJECT_MARKERS = [
    r"\b(mi\s+(for|while|if|clase|método|función|variable|array|lista))\b",
    r"\b(el\s+(for|while|if)\s+de\s+arriba)\b",
    r"\b(esta\s+(función|variable|clase|método|línea))\b",
    r"\b(el\s+que\s+(escribí|puse|tengo|estoy\s+usando))\b",
]


# ═══════════════════════════════════════════════════════════════════════
# DATACLASS DE RESULTADO PRAGMÁTICO
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PragmaticAnalysis:
    """
    Resultado del análisis pragmático sobre un prompt.
    Se combina con CognitiveAnalysis para producir el nivel Bloom final.
    """
    # Puntuaciones por heurística (-1 = baja abstracción, +1 = alta abstracción)
    h1_indexical_symbolic: float        # -1.0 indexical → +1.0 simbólico
    h2_symptom_mechanism: float         # -1.0 síntoma → +1.0 mecanismo
    h3_correction_comprehension: float  # -1.0 corrección → +1.0 comprensión
    h4_concrete_abstract: float         # -1.0 concreto → +1.0 abstracto

    # Score compuesto (-1.0 a +1.0)
    pragmatic_score: float

    # Ajuste de nivel recomendado (-2 a +1)
    level_adjustment: int

    # Confianza en el ajuste (0.0-1.0)
    adjustment_confidence: float

    # Etiqueta para el docente
    pragmatic_label: str    # "troubleshooting" | "conceptual" | "neutral"

    # Descripción del ajuste para debug
    rationale: str

    # Evidencias específicas detectadas
    evidence: list


# ═══════════════════════════════════════════════════════════════════════
# ANALIZADOR PRAGMÁTICO
# ═══════════════════════════════════════════════════════════════════════

class PragmaticAnalyzer:
    """
    Capa de análisis pragmático que refina la clasificación de Bloom
    de CognitiveAnalyzer.

    Integración:
        cognitive = CognitiveAnalyzer()
        pragmatic = PragmaticAnalyzer()

        bloom_raw = cognitive.analyze(prompt)
        bloom_refined = pragmatic.refine(bloom_raw, prompt)

    El método refine() devuelve (bloom_level_final, confidence_final, pragmatic_analysis).
    El nivel final puede ser igual, superior o inferior al detectado por keywords.
    """

    def analyze(self, prompt: str) -> PragmaticAnalysis:
        """
        Analiza las dimensiones pragmáticas de un prompt.
        Devuelve un PragmaticAnalysis con scores por heurística y ajuste recomendado.
        """
        text = prompt.lower().strip()
        evidence = []

        # ── H1: Indexical vs Simbólico ──
        n_indexical = sum(1 for p in INDEXICAL_MARKERS if re.search(p, text))
        n_symbolic  = sum(1 for p in SYMBOLIC_MARKERS  if re.search(p, text))

        if n_indexical > 0:
            evidence.append(f"H1 indexical: {n_indexical} marcadores → referencia concreta")
        if n_symbolic > 0:
            evidence.append(f"H1 simbólico: {n_symbolic} marcadores → relación abstracta")

        h1 = self._normalize(n_symbolic - n_indexical, scale=3.0)

        # ── H2: Síntoma vs Mecanismo ──
        n_symptom   = sum(1 for p in SYMPTOM_FOCUS_MARKERS   if re.search(p, text))
        n_mechanism = sum(1 for p in MECHANISM_FOCUS_MARKERS if re.search(p, text))

        if n_symptom > 0:
            evidence.append(f"H2 síntoma: {n_symptom} marcadores → troubleshooting")
        if n_mechanism > 0:
            evidence.append(f"H2 mecanismo: {n_mechanism} marcadores → comprensión causal")

        h2 = self._normalize(n_mechanism - n_symptom, scale=3.0)

        # ── H3: Corrección vs Comprensión ──
        n_correction    = sum(1 for p in CORRECTION_SEEKING_MARKERS if re.search(p, text))
        n_comprehension = sum(1 for p in MECHANISM_FOCUS_MARKERS    if re.search(p, text))

        if n_correction > 0:
            evidence.append(f"H3 corrección: {n_correction} marcadores → acción reparadora")

        h3 = self._normalize(n_comprehension - n_correction, scale=3.0)

        # ── H4: Objeto Concreto vs Abstracto ──
        n_concrete = sum(1 for p in CONCRETE_OBJECT_MARKERS if re.search(p, text))
        n_abstract = sum(1 for p in ABSTRACT_OBJECT_MARKERS if re.search(p, text))

        if n_concrete > 0:
            evidence.append(f"H4 concreto: {n_concrete} marcadores → objeto inmediato")
        if n_abstract > 0:
            evidence.append(f"H4 abstracto: {n_abstract} marcadores → concepto abstracto")

        h4 = self._normalize(n_abstract - n_concrete, scale=3.0)

        # ── Score compuesto ──
        # Pesos: H1 y H2 son los más discriminativos según la literatura
        pragmatic_score = (
            0.35 * h1 +
            0.30 * h2 +
            0.20 * h3 +
            0.15 * h4
        )

        # ── Ajuste de nivel y confianza ──
        level_adj, confidence, label, rationale = self._compute_adjustment(
            pragmatic_score, h1, h2, h3, h4, evidence
        )

        return PragmaticAnalysis(
            h1_indexical_symbolic=round(h1, 3),
            h2_symptom_mechanism=round(h2, 3),
            h3_correction_comprehension=round(h3, 3),
            h4_concrete_abstract=round(h4, 3),
            pragmatic_score=round(pragmatic_score, 3),
            level_adjustment=level_adj,
            adjustment_confidence=round(confidence, 2),
            pragmatic_label=label,
            rationale=rationale,
            evidence=evidence if evidence else ["Sin marcadores pragmáticos detectados."],
        )

    def refine(
        self,
        bloom_level: int,
        bloom_confidence: float,
        prompt: str,
    ) -> tuple[int, float, "PragmaticAnalysis"]:
        """
        Refina un nivel de Bloom detectado por keywords aplicando
        el análisis pragmático.

        Args:
            bloom_level: nivel Bloom raw de CognitiveAnalyzer (1-6)
            bloom_confidence: confianza del clasificador (0.0-1.0)
            prompt: texto original del prompt

        Returns:
            (bloom_level_final, confidence_final, pragmatic_analysis)
        """
        pragmatic = self.analyze(prompt)

        # Solo aplicar ajuste si la confianza pragmática es suficiente
        # y el ajuste tiene sentido dado el nivel actual
        if pragmatic.adjustment_confidence < 0.3:
            return bloom_level, bloom_confidence, pragmatic

        # No bajar de nivel 1 ni subir de nivel 6
        adjusted = max(1, min(6, bloom_level + pragmatic.level_adjustment))

        # La confianza final es la media ponderada:
        # si el ajuste es fuerte, penalizamos la confianza del clasificador
        if pragmatic.level_adjustment != 0:
            # Hubo ajuste: mezclar confianzas
            confidence_final = (
                0.6 * pragmatic.adjustment_confidence +
                0.4 * bloom_confidence
            )
        else:
            confidence_final = bloom_confidence

        return adjusted, round(confidence_final, 2), pragmatic

    # ──────────────────────────────────────────────────────────────────
    # HELPERS INTERNOS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(raw: float, scale: float = 3.0) -> float:
        """Normaliza un valor bruto de conteos al rango [-1, +1]."""
        return max(-1.0, min(1.0, raw / scale))

    @staticmethod
    def _compute_adjustment(
        score: float,
        h1: float, h2: float, h3: float, h4: float,
        evidence: list,
    ) -> tuple[int, float, str, str]:
        """
        Calcula el ajuste de nivel, la confianza, la etiqueta y la descripción.

        Reglas de decisión:
        - Score muy negativo (< -0.5): reducir 1 o 2 niveles (troubleshooting disfrazado de análisis)
        - Score muy positivo (> 0.5): aumentar 1 nivel (análisis genuino subestimado)
        - Score central: sin ajuste

        La reducción de 2 niveles solo cuando TODAS las señales apuntan
        en la misma dirección (confianza muy alta).
        """
        all_negative = h1 < -0.3 and h2 < -0.3 and h3 < -0.3
        all_positive = h1 > 0.3 and h2 > 0.3

        if score < -0.60 and all_negative:
            return (
                -2,
                min(abs(score), 0.9),
                "troubleshooting",
                "Todas las dimensiones pragmáticas señalan referencia indexical, foco en síntoma "
                "y búsqueda de corrección. El 'por qué' es de troubleshooting — APLICAR (Bloom 3), "
                "no ANALIZAR (Bloom 4). Ajuste: -2 niveles. (Austin, 1962; Bernstein, 1971)"
            )
        elif score < -0.28:
            return (
                -1,
                abs(score) * 0.85,
                "troubleshooting",
                "Señales pragmáticas predominantemente indexicales y orientadas a corrección. "
                "El nivel de Bloom está probablemente sobreestimado en 1 nivel. Ajuste: -1."
            )
        elif score > 0.45 and all_positive:
            return (
                +1,
                score * 0.85,
                "conceptual",
                "Señales pragmáticas de orientación simbólica y foco en mecanismo. "
                "El clasificador puede estar infraestimando el nivel de Bloom. Ajuste: +1. "
                "(Wineburg, 1991; Chi & Wylie, 2014)"
            )
        else:
            return (
                0,
                1.0 - abs(score),  # alta confianza cuando el score es neutro
                "neutral",
                "Señales pragmáticas no concluyentes o equilibradas. "
                "Se mantiene el nivel detectado por el clasificador de keywords."
            )


# ═══════════════════════════════════════════════════════════════════════
# WRAPPER DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════

class EnhancedBloomClassifier:
    """
    Combinación de CognitiveAnalyzer + PragmaticAnalyzer.
    Drop-in replacement para cualquier lugar donde se use CognitiveAnalyzer.analyze().
    """

    def __init__(self):
        # Import local para no romper si cognitive_analyzer no está en path
        try:
            from cognitive_analyzer import CognitiveAnalyzer
            self._cognitive = CognitiveAnalyzer()
        except ImportError:
            self._cognitive = None
        self._pragmatic = PragmaticAnalyzer()

    def analyze(self, prompt: str) -> dict:
        """
        Análisis completo: Bloom por keywords + refinado pragmático.

        Returns dict compatible con el formato de CognitiveAnalysis + extensiones:
          - bloom_level_raw: el nivel antes del refinado pragmático
          - bloom_level_final: el nivel tras el refinado
          - pragmatic_label: "troubleshooting" | "conceptual" | "neutral"
          - pragmatic_evidence: lista de marcadores detectados
          - level_was_adjusted: bool
          - adjustment_rationale: explicación del ajuste
        """
        if self._cognitive:
            raw = self._cognitive.analyze(prompt)
            level_raw = raw.bloom_level
            confidence_raw = raw.confidence
        else:
            # Fallback si no hay cognitive_analyzer: usar nivel 2 con baja confianza
            level_raw = 2
            confidence_raw = 0.3
            raw = None

        level_final, confidence_final, pragmatic = self._pragmatic.refine(
            level_raw, confidence_raw, prompt
        )

        return {
            "bloom_level_raw": level_raw,
            "bloom_level_final": level_final,
            "bloom_confidence_final": confidence_final,
            "level_was_adjusted": level_final != level_raw,
            "adjustment": level_final - level_raw,
            "pragmatic_label": pragmatic.pragmatic_label,
            "pragmatic_score": pragmatic.pragmatic_score,
            "pragmatic_evidence": pragmatic.evidence,
            "adjustment_rationale": pragmatic.rationale,
            "h1_indexical_symbolic": pragmatic.h1_indexical_symbolic,
            "h2_symptom_mechanism": pragmatic.h2_symptom_mechanism,
            "raw_analysis": raw,
        }


# ═══════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """
    Ejecuta los pares de prompts del test_diferencia documentado.
    Muestra cómo el parche pragmático corrige la clasificación donde
    el clasificador de keywords falla.
    """
    from cognitive_pragmatics import PragmaticAnalyzer

    analyzer = PragmaticAnalyzer()

    test_cases = [
        # (prompt, nivel_raw_simulado, nivel_esperado_tras_refine, descripción)
        (
            "¿por qué no me funciona este código? tengo un error en la línea 5",
            4,  # el clasificador de keywords ve "por qué" → le asigna ANALIZAR
            3,  # pero es troubleshooting → debería ser APLICAR
            "¿Por qué + error concreto? Troubleshooting, no análisis."
        ),
        (
            "¿por qué el bucle for es más eficiente que while para recorrer arrays?",
            4,  # clasificador ve "por qué" + "eficiente" → ANALIZAR
            4,  # correcto — aquí SÍ es análisis genuino
            "¿Por qué + abstracción? Análisis genuino. Sin cambio."
        ),
        (
            "mi función no compila, ¿qué está mal?",
            2,  # clasificador ve "qué está mal" → COMPRENDER
            2,  # debería ser APLICAR (troubleshooting) — pero sin el "por qué" ambiguo no hay mucho que ajustar
            "Diagnosis de error en instancia concreta."
        ),
        (
            "¿cuál es la diferencia conceptual entre recursión e iteración? ¿cuándo es mejor cada una?",
            4,  # ANALIZAR por keywords
            4,  # correcto, confirmado por pragmática
            "Pregunta abstracta con objeto simbólico. Sin ajuste."
        ),
        (
            "¿por qué me da null pointer exception cuando llamo a mi método?",
            4,  # "por qué" → ANALIZAR (error del clasificador)
            3,  # indexical + síntoma + corrección → APLICAR
            "¿Por qué + síntoma concreto? Clásico falso positivo Bloom 4."
        ),
        (
            "¿por qué existe el garbage collector y cuándo podría ser un problema?",
            4,  # ANALIZAR
            4,  # o quizás 5 — análisis genuino de mecanismo
            "¿Por qué existe + consecuencias? Análisis auténtico."
        ),
        (
            "¿por qué no me compila? lo tengo igual que el ejemplo",
            4,  # "por qué no" → ANALIZAR (error de keywords)
            2,  # es COMPRENDER a lo sumo — no hay razonamiento propio
            "Síntoma + referencia deíctica 'igual que'. Debería bajar."
        ),
    ]

    print("=" * 70)
    print("DEMO: CAPA DE ANÁLISIS PRAGMÁTICO")
    print("Fundamento: Austin (1962), Bernstein (1971), Wineburg (1991)")
    print("=" * 70)

    n_adjusted = 0
    n_correct = 0

    for prompt, level_raw, level_expected, desc in test_cases:
        level_final, confidence, prag = analyzer.refine(level_raw, 0.5, prompt)
        adjusted = level_final != level_raw
        correct = level_final == level_expected

        if adjusted:
            n_adjusted += 1
        if correct:
            n_correct += 1

        status = "✅" if correct else "⚠️"
        adj_str = f"Bloom {level_raw}→{level_final}" if adjusted else f"Bloom {level_raw} (sin cambio)"

        print(f"\n{status} {desc}")
        print(f"   Prompt: \"{prompt[:70]}{'...' if len(prompt)>70 else ''}\"")
        print(f"   {adj_str} | etiqueta: {prag.pragmatic_label} | score: {prag.pragmatic_score:+.2f}")
        if adjusted:
            print(f"   Ajuste: {prag.rationale[:90]}...")
        if prag.evidence[0] != "Sin marcadores pragmáticos detectados.":
            print(f"   Evidencias: {'; '.join(prag.evidence[:2])}")

    print(f"\n{'='*70}")
    print(f"Prompts ajustados: {n_adjusted}/{len(test_cases)}")
    print(f"Predicciones correctas: {n_correct}/{len(test_cases)}")
    print(f"Precisión: {n_correct/len(test_cases):.0%}")
    print("\nNota: el objetivo no es 100% sino reducir los falsos positivos")
    print("en casos de 'por qué' ambiguo sin alterar los genuinamente analíticos.")


if __name__ == "__main__":
    _demo()
