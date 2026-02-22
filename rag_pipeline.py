"""
PIPELINE RAG — Retrieval-Augmented Generation
==============================================
Replica la arquitectura del TFG de Pablo de Arriba:
  PDF → chunking → embeddings → ChromaDB → retrieval coseno → prompt → LLM

Simplificado para prototipo rápido. En producción:
- Re-ranking con cross-encoder
- Chunking semántico (no por caracteres)
- Evaluación con RAGAS (faithfulness, relevance, context recall)
"""

import os
import hashlib
from typing import Optional


def get_rag_pipeline(use_openai: bool = True):
    """
    Factory que devuelve el pipeline RAG configurado.
    Soporta dos modos:
      - OpenAI embeddings + ChromaDB (requiere API key)
      - Modo fallback sin embeddings (keyword matching)
    """
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAIRAGPipeline()
    else:
        return SimpleRAGPipeline()


class SimpleRAGPipeline:
    """
    Pipeline RAG simplificado SIN dependencias externas.
    Usa TF-IDF básico para retrieval. Perfecto para demo sin API key.
    """

    def __init__(self):
        self.documents: list[dict] = []  # {id, text, source, chunk_index}
        self.is_loaded = False

    def ingest_text(self, text: str, source: str = "documento", chunk_size: int = 500, overlap: int = 100):
        """Ingesta texto crudo → chunking → almacena."""
        chunks = self._chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()[:12]
            self.documents.append({
                "id": doc_id,
                "text": chunk,
                "source": source,
                "chunk_index": i,
            })
        self.is_loaded = True
        return len(chunks)

    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 100):
        """Ingesta PDF → extrae texto → chunking."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            source = os.path.basename(pdf_path)
            return self.ingest_text(full_text, source, chunk_size, overlap)
        except ImportError:
            # Fallback sin PyMuPDF
            try:
                import subprocess
                result = subprocess.run(
                    ["pdftotext", pdf_path, "-"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    source = os.path.basename(pdf_path)
                    return self.ingest_text(result.stdout, source, chunk_size, overlap)
            except FileNotFoundError:
                pass
            return 0

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieval por TF-IDF simplificado (coseno sobre keywords)."""
        if not self.documents:
            return []

        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            # Jaccard similarity
            intersection = query_words & doc_words
            union = query_words | doc_words
            score = len(intersection) / len(union) if union else 0
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "text": doc["text"],
                "source": doc["source"],
                "chunk_index": doc["chunk_index"],
                "score": round(score, 3),
            }
            for score, doc in scored[:top_k]
            if score > 0.01
        ]

    def build_context(self, query: str, top_k: int = 3) -> str:
        """Construye el contexto RAG para inyectar en el prompt."""
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Fragmento {i} — {r['source']}, sección {r['chunk_index']}]\n{r['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Chunking por caracteres con overlap.
        
        NOTA: Este es el chunking más simple posible. En producción hay 3 alternativas
        que probé mentalmente (y descarto temporalmente por orden de complejidad):
        
        1. Chunking semántico (LangChain SemanticChunker): corta en fronteras de
           significado usando embeddings. Mejor retrieval, pero añade latencia y
           dependencia de API en la ingesta. Para materiales de programación donde
           cada sección tiene headers claros, el beneficio marginal es dudoso.
        
        2. Chunking por secciones markdown/HTML: si el PDF viene de slides con
           estructura, parsear headers y cortar por sección. El TFG de Pablo usa
           PDFs de la asignatura que probablemente tienen estructura. Pero no puedo
           asumir formato sin ver los materiales reales.
        
        3. Recursive character splitter (LangChain): intenta cortar en párrafos,
           luego en frases, luego en palabras. Más robusto que mi implementación
           pero la diferencia es marginal para chunk_size=500.
        
        Mi implementación busca el último punto o salto de línea para no cortar
        mid-sentence. Es un 80/20 razonable.
        
        TODO (Fase A): evaluar con RAGAS (context_recall, faithfulness) si el chunking
        semántico mejora la calidad de retrieval en los materiales REALES de la asignatura.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # Intentar cortar en punto o salto de línea
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                cut_point = max(last_period, last_newline)
                if cut_point > chunk_size * 0.5:
                    chunk = chunk[:cut_point + 1]
                    end = start + cut_point + 1
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else end
        return chunks

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.documents),
            "sources": list(set(d["source"] for d in self.documents)),
            "avg_chunk_length": round(
                sum(len(d["text"]) for d in self.documents) / max(len(self.documents), 1)
            ),
        }


class OpenAIRAGPipeline(SimpleRAGPipeline):
    """
    Pipeline RAG con OpenAI embeddings + ChromaDB.
    Más preciso que el SimpleRAG. Requiere:
      - pip install chromadb openai
      - OPENAI_API_KEY en env
    """

    def __init__(self):
        super().__init__()
        self.collection = None
        self._setup_chromadb()

    def _setup_chromadb(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self.client = chromadb.Client()  # in-memory para prototipo
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
            self.collection = self.client.get_or_create_collection(
                name="course_materials",
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            print("⚠️ ChromaDB no instalado. Usando retrieval simple.")
            self.collection = None

    def ingest_text(self, text: str, source: str = "documento", chunk_size: int = 500, overlap: int = 100):
        """Ingesta con embeddings OpenAI en ChromaDB."""
        count = super().ingest_text(text, source, chunk_size, overlap)

        if self.collection is not None:
            # Añadir a ChromaDB con embeddings
            docs_to_add = self.documents[-count:]  # solo los nuevos
            self.collection.add(
                documents=[d["text"] for d in docs_to_add],
                ids=[d["id"] for d in docs_to_add],
                metadatas=[{"source": d["source"], "chunk_index": d["chunk_index"]} for d in docs_to_add],
            )
        return count

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieval semántico con ChromaDB (coseno sobre embeddings)."""
        if self.collection is None or self.collection.count() == 0:
            return super().retrieve(query, top_k)

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "?"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "score": round(1 - results["distances"][0][i], 3) if results["distances"] else 0,
            })
        return retrieved


# ──────────────────────────────────────────────
# CONTENIDO DE EJEMPLO PARA DEMO
# ──────────────────────────────────────────────

SAMPLE_COURSE_CONTENT = """
# Fundamentos de Programación — Materiales del Curso

## Tema 1: Variables y Tipos de Datos

Una variable es un espacio de memoria que almacena un valor. En Java, cada variable 
tiene un tipo que determina qué valores puede contener.

Tipos primitivos:
- int: números enteros (ej: int edad = 25;)
- double: números decimales (ej: double precio = 19.99;)
- boolean: verdadero o falso (ej: boolean activo = true;)
- char: un carácter (ej: char letra = 'A';)
- String: cadena de texto (ej: String nombre = "Ana";)

La declaración de variables sigue el patrón: tipo nombre = valor;

## Tema 2: Estructuras de Control — Condicionales

La estructura if-else permite ejecutar código según una condición:

if (condición) {
    // se ejecuta si la condición es verdadera
} else {
    // se ejecuta si la condición es falsa
}

Para múltiples condiciones, se usa if-else if-else:

if (nota >= 9) {
    System.out.println("Sobresaliente");
} else if (nota >= 7) {
    System.out.println("Notable");
} else if (nota >= 5) {
    System.out.println("Aprobado");
} else {
    System.out.println("Suspenso");
}

El operador ternario es una forma compacta: resultado = (condición) ? valorSi : valorNo;

## Tema 3: Bucles

El bucle for se usa cuando conocemos el número de iteraciones:

for (int i = 0; i < 10; i++) {
    System.out.println(i);
}

El bucle while se usa cuando la condición de parada es dinámica:

int contador = 0;
while (contador < limite) {
    // código
    contador++;
}

El bucle do-while garantiza al menos una ejecución:

do {
    // código
} while (condición);

IMPORTANTE: Cuidado con los bucles infinitos. Siempre asegúrate de que 
la condición de parada se alcanzará eventualmente.

## Tema 4: Arrays

Un array es una estructura que almacena múltiples valores del mismo tipo:

int[] numeros = new int[5];  // array de 5 enteros
numeros[0] = 10;             // asignar valor al primer elemento
numeros[1] = 20;

Recorrer un array con for:
for (int i = 0; i < numeros.length; i++) {
    System.out.println(numeros[i]);
}

Recorrer con for-each:
for (int n : numeros) {
    System.out.println(n);
}

Los arrays tienen tamaño fijo. Para tamaño dinámico, usar ArrayList.

## Tema 5: Funciones (Métodos)

Un método encapsula un bloque de código reutilizable:

public static int sumar(int a, int b) {
    return a + b;
}

Componentes de un método:
- Modificador de acceso (public, private)
- Tipo de retorno (int, void, String...)
- Nombre del método
- Parámetros entre paréntesis
- Cuerpo entre llaves

Llamada al método: int resultado = sumar(3, 5);

Los métodos void no devuelven valor:
public static void saludar(String nombre) {
    System.out.println("Hola, " + nombre);
}

## Tema 6: Recursión

Un método recursivo es aquel que se llama a sí mismo. Requiere:
1. Caso base: condición que detiene la recursión
2. Caso recursivo: llamada al propio método con parámetros modificados

Ejemplo — Factorial:
public static int factorial(int n) {
    if (n <= 1) return 1;        // caso base
    return n * factorial(n - 1);  // caso recursivo
}

Ejemplo — Fibonacci:
public static int fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

PRECAUCIÓN: La recursión consume memoria de pila (stack). 
Para valores grandes, considerar iteración como alternativa.
"""
