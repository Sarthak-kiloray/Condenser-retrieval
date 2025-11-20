"""Pydantic schemas for API requests and responses."""

from typing import List

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request schema for search endpoint."""

    query: str
    k: int = 5


class SearchResult(BaseModel):
    """Schema for a single search result."""

    doc_id: str
    text: str
    score: float


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str

