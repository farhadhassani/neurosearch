from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Product(BaseModel):
    """Represents a product in the catalog."""
    product_id: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    features: List[str] = Field(default_factory=list)
    semantic_id: Optional[str] = None  # e.g., "3 9 1"

class Query(BaseModel):
    """Represents a user query with optional filters."""
    text: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10

class SearchResult(BaseModel):
    """Represents a single search result."""
    product_id: str
    score: float
    product: Optional[Product] = None
    explanation: Optional[str] = None
    source: str = "unknown"  # dense, sparse, or generative

class SearchResponse(BaseModel):
    """Represents the final response to a search query."""
    query: str
    results: List[SearchResult]
    total_found: int = 0
