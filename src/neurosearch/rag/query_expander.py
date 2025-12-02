import openai

class QueryExpander:
    """Uses an LLM to expand a user query and optionally generate a HyDE (hypothetical document).
    This is a placeholder implementation – replace `openai.ChatCompletion` with your preferred LLM API.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.model = model
        if api_key:
            openai.api_key = api_key

    def expand(self, query: str) -> str:
        """Return an expanded version of the query.
        Simple prompt that asks the LLM to rewrite the query with more context.
        """
        prompt = f"Rewrite the following search query to be more detailed and include possible user intent: \"{query}\""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def extract_filters(self, query: str) -> dict:
        """Extract structured filters from the query (e.g., price, brand).
        Returns a dictionary of filters.
        """
        prompt = f"Extract filters from this query: \"{query}\". Return JSON with keys 'price_max', 'brand', 'category'. Return empty JSON if none."
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        import json
        try:
            return json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            return {}

    def hyde(self, query: str) -> str:
        """Generate a hypothetical document (HyDE) for the query.
        This is a stub – actual implementation would generate a passage that the query might retrieve.
        """
        prompt = f"Write a short paragraph that could answer the search query: \"{query}\""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
