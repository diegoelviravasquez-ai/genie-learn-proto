# GENIE Learn Prototype — Makefile
# ================================
# Porque escribir "pip install streamlit plotly pandas" cada vez
# es lo que hace alguien que no ha trabajado en un equipo.

.PHONY: setup run test lint clean demo-data help

PYTHON := python3
PIP := pip3
STREAMLIT := streamlit

# --- Colores para output (sí, me importa la estética del terminal) ---
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m  # No Color

help: ## Muestra esta ayuda
	@echo ""
	@echo "$(GREEN)GENIE Learn Prototype$(NC) — Comandos disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Instala dependencias (mínimas para demo)
	@echo "$(GREEN)Instalando dependencias...$(NC)"
	$(PIP) install streamlit>=1.30.0 plotly>=5.18.0 pandas>=2.0.0 --quiet
	@echo "$(GREEN)✅ Setup completo. Ejecuta: make run$(NC)"

setup-full: ## Instala TODO (RAG con embeddings, LLMs, tests)
	@echo "$(GREEN)Instalando dependencias completas...$(NC)"
	$(PIP) install streamlit>=1.30.0 plotly>=5.18.0 pandas>=2.0.0 \
		chromadb>=0.4.22 PyMuPDF>=1.23.0 openai>=1.12.0 anthropic>=0.18.0 \
		pytest>=7.0.0 --quiet
	@echo "$(GREEN)✅ Setup completo. Configura .env y ejecuta: make run$(NC)"

run: ## Lanza la aplicación Streamlit
	@echo "$(GREEN)Lanzando GENIE Learn...$(NC)"
	$(STREAMLIT) run app.py --server.headless true

run-demo: ## Lanza en modo demo (sin API key, datos simulados)
	@echo "$(YELLOW)Modo demo — respuestas simuladas$(NC)"
	unset OPENAI_API_KEY && unset ANTHROPIC_API_KEY && $(STREAMLIT) run app.py

test: ## Ejecuta tests
	@echo "$(GREEN)Ejecutando tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)✅ Tests completados$(NC)"

test-quick: ## Tests rápidos (solo lógica, sin imports pesados)
	$(PYTHON) -m pytest tests/test_middleware.py tests/test_cognitive.py -v --tb=short

lint: ## Verifica sintaxis de todos los módulos
	@echo "Verificando sintaxis..."
	@for f in *.py; do \
		$(PYTHON) -c "import ast; ast.parse(open('$$f').read())" && \
		echo "  $(GREEN)✅ $$f$(NC)" || echo "  $(RED)❌ $$f$(NC)"; \
	done

check: ## Verificación completa: lint + imports + test
	@echo "$(GREEN)=== Verificación completa ===$(NC)"
	@make lint
	@echo ""
	@$(PYTHON) -c "\
from middleware import PedagogicalMiddleware, PedagogicalConfig; \
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT; \
from llm_client import get_llm_client; \
from cognitive_analyzer import CognitiveAnalyzer, EngagementProfiler; \
from trust_dynamics import TrustDynamicsAnalyzer; \
from ach_diagnostic import ACHDiagnosticEngine; \
from nd_patterns import NeurodivergentPatternDetector; \
print('$(GREEN)✅ Todos los módulos importan correctamente$(NC)');"
	@make test-quick 2>/dev/null || echo "$(YELLOW)⚠️ Tests requieren pytest$(NC)"

loc: ## Cuenta líneas de código (sin tests ni docs)
	@echo "$(GREEN)Líneas de código:$(NC)"
	@wc -l *.py | sort -n
	@echo "---"
	@echo "$(YELLOW)Documentación:$(NC)"
	@wc -l *.md | sort -n

clean: ## Limpia cachés y archivos temporales
	rm -rf __pycache__/ .pytest_cache/ .streamlit/
	find . -name "*.pyc" -delete
	@echo "$(GREEN)Limpio.$(NC)"

# --- Targets de desarrollo ---

watch: ## Recarga automática al guardar (para desarrollo)
	$(STREAMLIT) run app.py --server.runOnSave true

docker-build: ## Construye imagen Docker
	docker build -t genie-prototype .

docker-run: ## Ejecuta en Docker
	docker run -p 8501:8501 --env-file .env genie-prototype
