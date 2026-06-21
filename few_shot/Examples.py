import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class SemanticExampleRetriever:
    """
    A retriever that uses semantic similarity to find the most relevant items in a dataset.
    """

    def __init__(self, json_file_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic retriever with data from a JSON file.

        Args:
            json_file_path: Path to the JSON file containing the data
            model_name: Name of the sentence-transformers model to use
        """
        self.data = self._load_json_data(json_file_path)['results']
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self._create_embeddings()

    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            raise Exception(f"Error loading JSON file: {e}")

    def _create_embeddings(self):
        """Create embeddings for all atomic facts in the dataset."""
        atomic_facts = [item["atomic_fact"] for item in self.data]
        self.embeddings = self.model.encode(atomic_facts)

    def search(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Search for items that semantically match the input query.

        Args:
            query: The query string to match against atomic facts
            top_n: Number of top matches to return

        Returns:
            List of matching items in the requested format
        """
        # Create embedding for the query
        query_embedding = self.model.encode(query)

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get indices of top matches
        top_indices = np.argsort(-similarities)[:top_n]

        results = []
        for idx in top_indices:
            item = self.data[idx]
            result = self._format_result(item)
            results.append(result)

        return results

    def _format_result(self, item: Dict[str, Any]) -> str:
        """Format a result item as a single string with the requested structure."""
        # Format evidence for output
        evidence_text = ""
        for evidence in item.get("evidence", []):
            if evidence.get("selected", False) or len(evidence_text) == 0:
                evidence_text += f"Title: {evidence.get('title', '')}\n"
                evidence_text += f"Text: {evidence.get('text', '')}\n\n"

        # Format history for output - handling list of lists format
        history_text = ""
        for entry in item.get("history", []):
            speaker = entry[0]
            message = entry[1]
            history_text += f"{speaker}: {message}\n"

        # Combine all parts into a single formatted string
        formatted_result = f"Evidence: {evidence_text.strip()}\n\n"
        formatted_result += f"Dialogue history: \n{history_text.strip()}\n"
        formatted_result += f"Speaker A: {item.get('A', '')}\n"
        formatted_result += f"Statement: {item.get('atomic_fact', '')}\n"
        formatted_result += f"{item.get('fact_check_result', '')}"

        return formatted_result


if __name__ == "__main__":

    # Initialize the retriever with your JSON file
    retriever = SemanticExampleRetriever("merged.json")

    # Search for a statement
    query = "NCAA Division II is the second-highest division of the NCAA."
    results = retriever.search("NCAA Division II is the second-highest division of the NCAA.", top_n=2)
    # Print the results
    for i, result_str in enumerate(results):
        print(f"Match {i + 1}:")
        print(result_str)
        print("-" * 80)