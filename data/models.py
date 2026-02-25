"""
data/models.py — Modelos SQLAlchemy para GENIE Learn
====================================================
Tablas: Student, Interaction, PedagogicalConfig.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    course_id = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    bloom_level = Column(Integer, nullable=False)  # 1-6
    copy_paste_score = Column(Float, nullable=False)  # 0-1
    scaffolding_mode = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class PedagogicalConfig(Base):
    __tablename__ = "pedagogical_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(String(255), nullable=False, index=True)
    socratic_mode = Column(Boolean, nullable=False, default=True)
    max_prompts = Column(Integer, nullable=False)
    hallucination_rate = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
