"""
CLIENTE LLM — Abstracción sobre múltiples proveedores
======================================================
Soporta:
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Anthropic (Claude Sonnet)
  - Mock mode (para demos sin API key)
"""

import os
import time
import json


def get_llm_client():
    """
    Factory: devuelve el cliente LLM disponible.
    Prioridad: ANTHROPIC_API_KEY → claude-sonnet-4-20250514,
               OPENAI_API_KEY → gpt-4o-mini,
               si ninguna existe → modo demo (mock).
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIClient()
    return MockLLMClient()


class MockLLMClient:
    """
    Cliente simulado para demos SIN API key.
    Genera respuestas coherentes basadas en el contexto RAG.
    """

    def __init__(self):
        self.model_name = "mock-demo"

    def chat(self, system_prompt: str, user_prompt: str, context: str = "") -> dict:
        """Simula una respuesta del LLM."""
        start = time.time()

        # Generar respuesta simulada basada en el scaffolding detectado
        if "NIVEL SOCRÁTICO" in system_prompt or "NIVEL PROGRESIVO 1/4" in system_prompt:
            response = self._socratic_response(user_prompt, context)
        elif "NIVEL PISTA" in system_prompt or "MODO PISTAS" in system_prompt or "NIVEL PROGRESIVO 2/4" in system_prompt:
            response = self._hint_response(user_prompt, context)
        elif "NIVEL EJEMPLO" in system_prompt or "MODO EJEMPLOS" in system_prompt or "NIVEL PROGRESIVO 3/4" in system_prompt:
            response = self._example_response(user_prompt, context)
        elif "NIVEL EXPLICACIÓN" in system_prompt or "NIVEL PROGRESIVO 4/4" in system_prompt:
            response = self._direct_response(user_prompt, context)
        elif "MODO ANALOGÍAS" in system_prompt:
            response = self._analogy_response(user_prompt, context)
        elif "MODO DIRECTO" in system_prompt:
            response = self._direct_response(user_prompt, context)
        elif "MODO DESAFÍO" in system_prompt:
            response = self._challenge_response(user_prompt, context)
        elif "MODO RUBBER DUCK" in system_prompt:
            response = self._rubber_duck_response(user_prompt, context)
        else:
            response = self._direct_response(user_prompt, context)

        elapsed = int((time.time() - start) * 1000)

        return {
            "response": response,
            "model": self.model_name,
            "tokens_used": len(response.split()),
            "response_time_ms": elapsed + 200,  # simular latencia
        }

    def _socratic_response(self, prompt: str, context: str) -> str:
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["bucle", "for", "while"]):
            return (
                "Interesante pregunta sobre bucles. Antes de darte la respuesta, "
                "me gustaría que reflexionaras sobre lo siguiente:\n\n"
                "1. ¿Sabes de antemano cuántas veces necesitas repetir la operación?\n"
                "2. Si la respuesta es sí, ¿qué tipo de bucle sería más apropiado?\n"
                "3. ¿Qué tres componentes necesita ese bucle para funcionar correctamente?\n\n"
                "Piensa en estas preguntas y dime qué se te ocurre. Estoy aquí para guiarte."
            )
        elif any(w in prompt_lower for w in ["array", "lista", "vector"]):
            return (
                "Antes de hablar de arrays, hagamos un ejercicio mental:\n\n"
                "Imagina que tienes que guardar las notas de 30 estudiantes. "
                "¿Crearías 30 variables separadas (nota1, nota2, nota3...)? "
                "¿Qué problemas tendría eso?\n\n"
                "Cuando me respondas, te guío hacia la solución."
            )
        elif any(w in prompt_lower for w in ["función", "método", "def"]):
            return (
                "Las funciones son uno de los conceptos más potentes en programación. "
                "Pero antes de explicártelo, reflexiona:\n\n"
                "¿Has tenido que escribir el mismo bloque de código varias veces en un programa? "
                "¿Qué pasaría si quisieras cambiar ese bloque — tendrías que cambiarlo en todos los sitios?\n\n"
                "¿Ves el problema? A partir de ahí entenderás para qué sirven las funciones."
            )
        else:
            return (
                "Buena pregunta. Pero antes de responderte directamente, "
                "me gustaría saber: ¿qué has intentado hasta ahora? "
                "¿Qué parte del problema entiendes y cuál te bloquea?\n\n"
                "Si me cuentas dónde estás, puedo orientarte mejor."
            )

    def _hint_response(self, prompt: str, context: str) -> str:
        if context:
            return (
                f"Bien, veo que lo has intentado. Te doy una pista:\n\n"
                f"En los materiales del curso hay un fragmento relevante que dice algo sobre "
                f"este tema. La clave está en cómo se estructura la sintaxis.\n\n"
                f"Pista concreta: revisa la sección sobre la declaración del elemento "
                f"que necesitas. Fíjate en el patrón: tipo + nombre + asignación.\n\n"
                f"¿Te ayuda esto o necesitas más orientación?"
            )
        return (
            "Te doy una pista sin resolver el problema por ti:\n\n"
            "Piensa en el problema como si tuvieras que explicárselo a alguien "
            "que no sabe programar. ¿Cuáles serían los pasos en lenguaje natural?\n\n"
            "Cuando los tengas, traduce cada paso a una línea de código."
        )

    def _example_response(self, prompt: str, context: str) -> str:
        return (
            "Veo que necesitas un ejemplo. Te pongo uno SIMILAR pero distinto "
            "a tu problema para que puedas extrapolarlo:\n\n"
            "```java\n"
            "// Ejemplo: calcular la media de un array\n"
            "int[] datos = {10, 20, 30, 40, 50};\n"
            "int suma = 0;\n"
            "for (int i = 0; i < datos.length; i++) {\n"
            "    suma += datos[i];\n"
            "}\n"
            "double media = (double) suma / datos.length;\n"
            "```\n\n"
            "Observa: (1) inicializamos un acumulador, (2) recorremos con for, "
            "(3) operamos en cada iteración, (4) calculamos el resultado final.\n\n"
            "¿Puedes aplicar este patrón a tu problema?"
        )

    def _analogy_response(self, prompt: str, context: str) -> str:
        return (
            "Te lo explico con una analogía del mundo real:\n\n"
            "Imagina que un array es como una fila de casilleros numerados. "
            "Cada casillero guarda un valor. Para acceder a uno concreto, usas su número (índice). "
            "¿Qué operación del día a día te recuerda a 'recorrer todos los casilleros'?\n\n"
            "Conecta esta analogía con tu pregunta y dime qué relación ves."
        )

    def _challenge_response(self, prompt: str, context: str) -> str:
        return (
            "Antes de atacar tu problema, intenta resolver esto más simple:\n\n"
            "**Desafío previo:** Dado un array de 5 números, calcula su suma. "
            "Usa un bucle for. ¿Puedes escribir el pseudocódigo o el código?\n\n"
            "Cuando lo intentes (aunque falle), te ayudo con tu problema original."
        )

    def _rubber_duck_response(self, prompt: str, context: str) -> str:
        return (
            "Vamos a hacer el ejercicio del patito de goma: explícame TU problema paso a paso, "
            "como si yo no supiera nada de programación.\n\n"
            "Empieza por: ¿qué quieres conseguir? ¿Qué has hecho hasta ahora? "
            "¿En qué momento exacto ocurre el error o te bloqueas?\n\n"
            "No te doy la solución — primero necesito entender cómo ves TÚ el problema."
        )

    def _direct_response(self, prompt: str, context: str) -> str:
        if context:
            return (
                f"Basándome en los materiales del curso, aquí tienes la explicación:\n\n"
                f"El concepto que preguntas está cubierto en el temario. "
                f"Los puntos clave son:\n\n"
                f"1. La sintaxis básica sigue el patrón que has visto en clase.\n"
                f"2. Es importante recordar las reglas de ámbito (scope).\n"
                f"3. Practica con los ejercicios del tema para consolidar.\n\n"
                f"Contexto relevante del curso:\n{context[:300]}..."
            )
        return (
            "Aquí tienes la explicación paso a paso:\n\n"
            "Este es un concepto fundamental que necesitas dominar. "
            "La clave está en entender cómo se relacionan las partes.\n\n"
            "Te recomiendo practicar con los ejercicios del tema."
        )


class OpenAIClient:
    """Cliente para API de OpenAI."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = "gpt-4o-mini"

    def chat(self, system_prompt: str, user_prompt: str, context: str = "") -> dict:
        start = time.time()

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            user_content = (
                f"CONTEXTO DEL CURSO (materiales proporcionados por el profesor):\n"
                f"---\n{context}\n---\n\n"
                f"PREGUNTA DEL ESTUDIANTE:\n{user_prompt}"
            )
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        elapsed = int((time.time() - start) * 1000)

        return {
            "response": response.choices[0].message.content,
            "model": self.model_name,
            "tokens_used": response.usage.total_tokens,
            "response_time_ms": elapsed,
        }


class AnthropicClient:
    """Cliente para API de Anthropic."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model_name = "claude-sonnet-4-20250514"

    def chat(self, system_prompt: str, user_prompt: str, context: str = "") -> dict:
        start = time.time()

        if context:
            user_content = (
                f"CONTEXTO DEL CURSO:\n---\n{context}\n---\n\n"
                f"PREGUNTA DEL ESTUDIANTE:\n{user_prompt}"
            )
        else:
            user_content = user_prompt

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        elapsed = int((time.time() - start) * 1000)

        return {
            "response": response.content[0].text,
            "model": self.model_name,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "response_time_ms": elapsed,
        }
