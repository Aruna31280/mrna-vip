"""Pydantic schemas: LLM report (FR-4.1) and API request bodies."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class SafetyReport(BaseModel):
    """Exactly the five fields required by FR-4.1 - nothing more, nothing less."""
    generated_at: str
    patient_summary: str
    model_output: dict
    guideline_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str

class PatientInput(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Biomarker_Level: float = Field(..., ge=0)
    Pre_Existing_Conditions: int = Field(..., ge=0, le=10)
    Vaccine_Dose_mcg: float = Field(..., gt=0)
    Prior_Reaction_History: Literal["None", "Mild", "Moderate"]


class QueryInput(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(3, ge=1, le=10)
