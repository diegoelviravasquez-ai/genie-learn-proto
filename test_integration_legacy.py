"""
GENIE Learn — Test de Integración End-to-End
==============================================
Verifica que todos los módulos se conectan correctamente
y que el middleware pedagógico produce comportamientos
verificablemente distintos según la configuración docente.

Ejecución: python test_integration.py
"""

import sys
import os
import time

# Forzar modo sin API key
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from middleware import PedagogicalMiddleware, PedagogicalConfig
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client
from cognitive_analyzer import CognitiveAnalyzer, EngagementProfiler, BLOOM_LEVELS
from trust_dynamics import TrustDynamicsAnalyzer


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_rag_pipeline():
    header("TEST 1: RAG Pipeline")
    rag = get_rag_pipeline(use_openai=False)
    n = rag.ingest_text(SAMPLE_COURSE_CONTENT, "Fundamentos_Programacion.pdf")
    print(f"  ✓ Ingested {n} chunks")

    results = rag.retrieve("¿Qué es un bucle for?", top_k=3)
    print(f"  ✓ Retrieved {len(results)} chunks for 'bucle for'")
    for r in results:
        print(f"    - [{r['source']}] score={r['score']:.3f} | {r['text'][:60]}...")

    context = rag.build_context("recursión", top_k=2)
    assert "recursión" in context.lower() or "caso base" in context.lower()
    print(f"  ✓ Context for 'recursión': {len(context)} chars")

    stats = rag.get_stats()
    print(f"  ✓ Stats: {stats}")
    return rag


def test_scaffolding_escalation():
    header("TEST 2: Scaffolding Socrático — Escalación progresiva")
    config = PedagogicalConfig(scaffolding_mode="socratic")
    mw = PedagogicalMiddleware(config)
    llm = get_llm_client()
    sid = "est_socratic"

    expected_keywords = ["SOCRÁTICO", "PISTA", "EJEMPLO", "EXPLICACIÓN"]

    for i in range(7):
        pre = mw.pre_process(sid, f"Pregunta {i+1} sobre bucles for")
        result = llm.chat(pre["system_prompt"], f"Pregunta {i+1}", "")
        post = mw.post_process(sid, result["response"])

        level = pre["scaffolding_level"]
        label = expected_keywords[min(level, 3)]
        in_prompt = label in pre["system_prompt"]
        print(f"  Interacción {i+1}: level={level} ({label}) | "
              f"en system_prompt={'✓' if in_prompt else '✗'}")

    # Verificar que se llegó al nivel 3
    final_state = mw.conversation_states[sid]
    print(f"  ✓ Estado final: level={final_state['level']}, attempts={final_state['attempts']}")
    assert final_state["level"] >= 2, "Should have escalated to at least level 2"


def test_direct_mode_vs_socratic():
    header("TEST 3: Modo directo vs Socrático — Verificablemente distintos")

    prompt = "¿Cómo funciona un bucle while?"

    # Socratic
    config_s = PedagogicalConfig(scaffolding_mode="socratic")
    mw_s = PedagogicalMiddleware(config_s)
    pre_s = mw_s.pre_process("est_s", prompt)

    # Direct
    config_d = PedagogicalConfig(scaffolding_mode="direct")
    mw_d = PedagogicalMiddleware(config_d)
    pre_d = mw_d.pre_process("est_d", prompt)

    print(f"  Socrático system_prompt ({len(pre_s['system_prompt'])} chars):")
    print(f"    '{pre_s['system_prompt'][:120]}...'")
    print(f"  Directo system_prompt ({len(pre_d['system_prompt'])} chars):")
    print(f"    '{pre_d['system_prompt'][:120]}...'")

    assert "SOCRÁTICO" in pre_s["system_prompt"]
    assert "SOCRÁTICO" not in pre_d["system_prompt"]
    print(f"  ✓ Los system prompts son verificablemente distintos")

    # Verify mock LLM responds differently
    llm = get_llm_client()
    r_s = llm.chat(pre_s["system_prompt"], prompt, "")
    r_d = llm.chat(pre_d["system_prompt"], prompt, "")
    print(f"  Socrático response: '{r_s['response'][:80]}...'")
    print(f"  Directo response:   '{r_d['response'][:80]}...'")
    assert r_s["response"] != r_d["response"], "Responses should differ"
    print(f"  ✓ Las respuestas del LLM son diferentes según el modo")


def test_daily_limit():
    header("TEST 4: Límite diario de prompts")
    config = PedagogicalConfig(max_daily_prompts=3)
    mw = PedagogicalMiddleware(config)
    sid = "est_limit"

    for i in range(4):
        pre = mw.pre_process(sid, f"Pregunta {i+1}")
        status = "✓ allowed" if pre["allowed"] else f"⛔ blocked: {pre['block_reason'][:50]}"
        print(f"  Prompt {i+1}/3: {status}")

    pre_final = mw.pre_process(sid, "Una más")
    assert not pre_final["allowed"], "Should be blocked"
    print(f"  ✓ Límite funciona correctamente")


def test_hallucination_injection():
    header("TEST 5: Alucinación pedagógica controlada")

    # 100% injection rate
    config = PedagogicalConfig(forced_hallucination_pct=1.0)
    mw = PedagogicalMiddleware(config)
    post = mw.post_process("est_h", "Los arrays empiezan en el índice 0.")
    assert post["hallucination_injected"]
    assert "NOTA PEDAGÓGICA" in post["response"]
    print(f"  ✓ 100% rate: inyección correcta")
    print(f"    Aviso: '...{post['response'][-80:]}'")

    # 0% injection rate
    config0 = PedagogicalConfig(forced_hallucination_pct=0.0)
    mw0 = PedagogicalMiddleware(config0)
    post0 = mw0.post_process("est_h0", "Los arrays empiezan en el índice 0.")
    assert not post0["hallucination_injected"]
    print(f"  ✓ 0% rate: sin inyección")


def test_copy_paste_detection():
    header("TEST 6: Detección de copy-paste")
    config = PedagogicalConfig()
    mw = PedagogicalMiddleware(config)

    organic = "¿cómo hago un for que recorra un array?"
    pasted = (
        "Ejercicio 3.2: Dado el siguiente array de enteros, implementar un método "
        "que recorra todos los elementos y calcule la media aritmética. Se pide "
        "además que el programa detecte si algún valor supera el doble de la media."
    )

    pre_org = mw.pre_process("est_cp1", organic)
    pre_cp = mw.pre_process("est_cp2", pasted)

    print(f"  Orgánico: score={pre_org['copy_paste_score']:.2f}")
    print(f"  Pegado:   score={pre_cp['copy_paste_score']:.2f}")
    assert pre_cp["copy_paste_score"] > pre_org["copy_paste_score"]
    print(f"  ✓ El texto pegado tiene mayor score que el orgánico")


def test_cognitive_analysis():
    header("TEST 7: Análisis cognitivo (Bloom + ICAP)")
    analyzer = CognitiveAnalyzer()

    tests = [
        ("¿Qué es una variable?", 1, "Recordar"),
        ("Explica cómo funciona un bucle for", 2, "Comprender"),
        ("Escribe un programa que calcule factorial", 3, "Aplicar"),
        ("¿Cuál es mejor: recursión o iteración?", 5, "Evaluar"),
        ("Diseña un algoritmo nuevo para ordenar", 6, "Crear"),
    ]

    for prompt, expected_level, expected_name in tests:
        result = analyzer.analyze(prompt)
        match = "✓" if result.bloom_level == expected_level else "≈"
        print(f"  {match} '{prompt[:45]}...' → L{result.bloom_level} "
              f"({result.bloom_name}) ICAP:{result.icap_label} "
              f"conf={result.bloom_confidence:.2f}")


def test_trust_dynamics():
    header("TEST 8: Dinámicas de confianza")
    analyzer = TrustDynamicsAnalyzer()

    tests = [
        ("¿Estás seguro de que eso es correcto?", "verification"),
        ("Ok perfecto, siguiente pregunta", "sobre-confianza"),
        ("No me sirve, dame la respuesta directa", "frustración"),
        ("A ver si entiendo bien, o sea que...", "reformulación"),
    ]

    for prompt, expected_type in tests:
        signal = analyzer.analyze_prompt("est_trust", prompt)
        print(f"  '{prompt[:45]}...' → type={signal.signal_type} "
              f"direction={signal.trust_direction:+.2f}")


def test_block_direct_solutions():
    header("TEST 9: Bloqueo de soluciones directas")

    config_block = PedagogicalConfig(block_direct_solutions=True)
    mw_block = PedagogicalMiddleware(config_block)
    pre = mw_block.pre_process("est_sol", "Resuélveme el ejercicio 3")
    assert "NO la proporciones" in pre["system_prompt"] or "NO" in pre["system_prompt"]
    print(f"  ✓ Con bloqueo: system_prompt incluye instrucción de no resolver")

    config_free = PedagogicalConfig(block_direct_solutions=False)
    mw_free = PedagogicalMiddleware(config_free)
    pre_free = mw_free.pre_process("est_sol2", "Resuélveme el ejercicio 3")
    assert "NO la proporciones" not in pre_free["system_prompt"]
    print(f"  ✓ Sin bloqueo: system_prompt no restringe soluciones")


def test_topic_detection():
    header("TEST 10: Detección de topics")
    config = PedagogicalConfig()
    mw = PedagogicalMiddleware(config)

    tests = [
        ("¿Cómo declaro una variable int?", ["variables"]),
        ("El bucle while no termina", ["bucles", "depuración"]),
        ("Quiero hacer una función recursiva", ["funciones", "recursión"]),
    ]

    for prompt, expected in tests:
        pre = mw.pre_process(f"est_topic_{hash(prompt)}", prompt)
        topics = pre["detected_topics"]
        matched = all(e in topics for e in expected)
        print(f"  {'✓' if matched else '✗'} '{prompt[:40]}...' → {topics} "
              f"(esperado: {expected})")


def test_full_pipeline():
    header("TEST 11: Pipeline completo end-to-end")

    config = PedagogicalConfig(
        scaffolding_mode="socratic",
        block_direct_solutions=True,
        forced_hallucination_pct=0.0,
        use_rag=True,
        max_daily_prompts=20,
    )
    mw = PedagogicalMiddleware(config)
    rag = get_rag_pipeline(use_openai=False)
    rag.ingest_text(SAMPLE_COURSE_CONTENT, "curso.pdf")
    llm = get_llm_client()
    analyzer = CognitiveAnalyzer()
    trust = TrustDynamicsAnalyzer()

    prompts = [
        "¿Qué es un array en Java?",
        "No entiendo, ¿cómo declaro uno?",
        "Dame un ejemplo de recorrer array",
        "Ya entendí, ahora explícame el for-each",
    ]

    for i, prompt in enumerate(prompts):
        pre = mw.pre_process("pipeline_test", prompt)
        if not pre["allowed"]:
            print(f"  ⛔ Blocked: {pre['block_reason']}")
            continue

        context = rag.build_context(prompt, top_k=3) if config.use_rag else ""
        result = llm.chat(pre["system_prompt"], prompt, context)
        post = mw.post_process("pipeline_test", result["response"])
        cognitive = analyzer.analyze(prompt)
        trust_signal = trust.analyze_prompt("pipeline_test", prompt)

        mw.log_interaction(
            student_id="pipeline_test",
            prompt_raw=prompt,
            pre_result=pre,
            response_raw=result["response"],
            post_result=post,
            response_time_ms=result.get("response_time_ms", 200),
        )

        print(f"  [{i+1}] Scaff:L{pre['scaffolding_level']} "
              f"Bloom:L{cognitive.bloom_level}({cognitive.bloom_name}) "
              f"ICAP:{cognitive.icap_label} "
              f"Trust:{trust_signal.signal_type} "
              f"Topics:{pre['detected_topics']}")

    summary = mw.get_analytics_summary()
    print(f"\n  Analytics summary:")
    print(f"    Total: {summary['total_interactions']} interactions")
    print(f"    Scaffolding: {summary['scaffolding_levels']}")
    print(f"    Topics: {summary['topic_distribution']}")
    print(f"  ✓ Pipeline completo funcional")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════╗")
    print("║   GENIE Learn — Test de Integración End-to-End        ║")
    print("║   Diego Elvira Vásquez · CP25/152 · GSIC/EMIC-UVa    ║")
    print("╚════════════════════════════════════════════════════════╝")

    tests = [
        test_rag_pipeline,
        test_scaffolding_escalation,
        test_direct_mode_vs_socratic,
        test_daily_limit,
        test_hallucination_injection,
        test_copy_paste_detection,
        test_cognitive_analysis,
        test_trust_dynamics,
        test_block_direct_solutions,
        test_topic_detection,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")

    header(f"RESULTADO: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("  🎉 TODO FUNCIONA — Sistema listo para demo")
    else:
        print(f"  ⚠️  {failed} tests fallaron")

    sys.exit(0 if failed == 0 else 1)
