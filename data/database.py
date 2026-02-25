"""
data/database.py — Engine, init, logging y consulta de interacciones
====================================================================
- get_engine(): DATABASE_URL desde .env o SQLite genie_demo.db
- init_db(): crea tablas
- log_interaction(): inserta una interacción
- get_interactions_df(): DataFrame pandas de interacciones
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Student, Interaction, PedagogicalConfig


def get_engine():
    """Engine desde DATABASE_URL en .env; si no existe, SQLite genie_demo.db."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    url = os.getenv("DATABASE_URL")
    if not url:
        url = "sqlite:///genie_demo.db"
    return create_engine(url, echo=False)


def init_db(engine=None):
    """Crea todas las tablas. Si no se pasa engine, usa get_engine()."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def log_interaction(
    student_id: int,
    prompt: str,
    response: str,
    bloom_level: int,
    copy_paste_score: float,
    scaffolding_mode: str,
    engine=None,
):
    """Guarda una fila en Interaction."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(
            Interaction(
                student_id=student_id,
                prompt=prompt,
                response=response,
                bloom_level=bloom_level,
                copy_paste_score=copy_paste_score,
                scaffolding_mode=scaffolding_mode,
            )
        )
        session.commit()


def get_interactions_df(course_id=None, engine=None):
    """DataFrame de pandas con todas las interacciones (opcionalmente filtrado por course_id)."""
    import pandas as pd

    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    with Session() as session:
        if course_id is not None:
            q = session.query(Interaction).join(Student, Interaction.student_id == Student.id).filter(Student.course_id == course_id)
        else:
            q = session.query(Interaction)
        rows = q.all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "id", "student_id", "prompt", "response",
                "bloom_level", "copy_paste_score", "scaffolding_mode", "timestamp",
            ]
        )
    return pd.DataFrame(
        [
            {
                "id": r.id,
                "student_id": r.student_id,
                "prompt": r.prompt,
                "response": r.response,
                "bloom_level": r.bloom_level,
                "copy_paste_score": r.copy_paste_score,
                "scaffolding_mode": r.scaffolding_mode,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]
    )
