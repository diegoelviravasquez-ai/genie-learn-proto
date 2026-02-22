# 🧬 GENIE Learn — Prototipo de Chatbot Educativo con IA Generativa

**Réplica funcional mínima** del sistema descrito en el paper LAK 2026 (Ortega-Arranz et al.) 
y el TFG de Pablo de Arriba Mendizábal (UVa, 2025).

Construido como demostración técnica para el contrato CP25/152, nodo UVa del proyecto GENIE Learn.

---

## ⚡ ARRANQUE RÁPIDO (2 minutos)

```bash
# 1. Instalar dependencias mínimas
pip install streamlit plotly pandas

# 2. Ejecutar (modo demo, sin API key necesaria)
streamlit run app.py
```

Se abre en `http://localhost:8501`. Funciona inmediatamente con respuestas simuladas.

---

## 🔑 CON LLM REAL (respuestas de GPT-4o o Claude)

```bash
# Opción A: OpenAI
pip install openai chromadb PyMuPDF
export OPENAI_API_KEY="sk-..."
streamlit run app.py

# Opción B: Anthropic
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

Con API key, el RAG usa embeddings OpenAI + ChromaDB (retrieval semántico real).
Sin API key, usa TF-IDF simplificado (funciona igual para demo).

---

## 🏗️ ARQUITECTURA (4 capas, como el sistema real)

```
┌─────────────────────────────────────────────────┐
│           CAPA 1: INTERFAZ                       │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  🎓 Estudiante│    │  🧑‍🏫 Docente          │   │
│  │  (chat)      │    │  (config + analytics)│   │
│  └──────┬───────┘    └──────────┬───────────┘   │
├─────────┼───────────────────────┼───────────────┤
│         ▼           CAPA 2: MIDDLEWARE           │
│  ┌─────────────────────────────────────────┐    │
│  │  Pedagogical Configurations Engine       │    │
│  │  • Límite diario de prompts              │    │
│  │  • Scaffolding socrático (4 niveles)     │    │
│  │  • Bloqueo de soluciones directas        │    │
│  │  • Alucinaciones pedagógicas controladas │    │
│  │  • Detección de copy-paste               │    │
│  │  • System prompt dinámico                │    │
│  └──────────────┬──────────────────────────┘    │
├─────────────────┼───────────────────────────────┤
│                 ▼      CAPA 3: LLM + RAG        │
│  ┌──────────────────┐  ┌────────────────────┐   │
│  │  LLM Client       │  │  RAG Pipeline      │   │
│  │  • OpenAI         │  │  • PDF → chunks    │   │
│  │  • Anthropic      │  │  • Embeddings      │   │
│  │  • Mock (demo)    │  │  • ChromaDB        │   │
│  └──────────────────┘  │  • Retrieval coseno │   │
│                         └────────────────────┘   │
├──────────────────────────────────────────────────┤
│               CAPA 4: ANALYTICS                  │
│  ┌─────────────────────────────────────────┐    │
│  │  GenAI Analytics Engine                  │    │
│  │  • Topics detection + auto-tagging       │    │
│  │  • Copy-paste scoring                    │    │
│  │  • Scaffolding level tracking            │    │
│  │  • Interaction logging                   │    │
│  │  • Dashboard con Plotly                  │    │
│  └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## 📂 ESTRUCTURA DE ARCHIVOS

```
genie_prototype/
├── app.py              # App Streamlit principal (3 vistas)
├── middleware.py        # Motor de reglas pedagógicas (la innovación clave)
├── rag_pipeline.py      # Pipeline RAG (Simple + OpenAI/ChromaDB)
├── llm_client.py        # Abstracción LLM (OpenAI, Anthropic, Mock)
├── requirements.txt     # Dependencias
└── README.md            # Este archivo
```

---

## 🎯 QUÉ DEMUESTRA ESTE PROTOTIPO

| Competencia requerida (CP25/152) | Cómo la demuestra el prototipo |
|----------------------------------|-------------------------------|
| APIs LLM | Clientes para OpenAI y Anthropic con fallback |
| RAG | Pipeline completo: ingesta PDF → chunking → embeddings → retrieval |
| Prompt engineering | System prompts dinámicos con inyección de configuraciones pedagógicas |
| Guardrails | Límites de uso, bloqueo de soluciones, detección de copy-paste |
| Learning Analytics | Dashboard con 5 visualizaciones + logging completo |
| HCAI / Agencia docente | Panel de configuración donde el DOCENTE decide, no el sistema |
| Arquitectura | 4 capas separadas, middleware como motor de reglas |

---

## 🔬 FUNDAMENTACIÓN TEÓRICA

- **Scaffolding socrático**: Wood, Bruner & Ross (1976). Zona de Desarrollo Próximo (Vygotsky)
- **Desirable difficulties**: Bjork (1994). Limitar prompts como fricción cognitiva productiva
- **Value-Sensitive Design**: Friedman et al. (2017). El docente configura, no el ingeniero
- **GenAI Analytics**: Ortega-Arranz et al. (LAK 2026). Monitorización de interacciones
- **HCAI**: Topali et al. (2024). Dimitriadis et al. (2021). 3 requisitos HC
- **RAG**: Lewis et al. (2020). Contextualización sin fine-tuning
- **DSRM**: Peffers et al. (2007). Metodología de investigación en diseño

---



---

*Diego Elvira Vásquez · Febrero 2026 · Prototipo para CP25/152*
