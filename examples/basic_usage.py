"""
Basic Usage Example: Chatterjee's Xi for Semantic Similarity

This example demonstrates the core functionality of computing Xi similarity
between sentence embeddings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.similarity import chatterjee_xi, symmetric_xi, EmbeddingModel
from src.similarity.metrics import cosine_similarity_score


def main():
    print("=" * 70)
    print("Chatterjee's Xi for Semantic Similarity - Basic Example")
    print("=" * 70)

    # Initialize embedding model
    print("\n1. Loading sentence embedding model...")
    model = EmbeddingModel('all-MiniLM-L6-v2')
    print("   ✓ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")

    # Example sentence pairs
    examples = [
        {
            'name': 'High Similarity (Paraphrase)',
            'sent1': "The quick brown fox jumps over the lazy dog.",
            'sent2': "A swift auburn fox leaps over a sleepy canine."
        },
        {
            'name': 'Medium Similarity (Related Topic)',
            'sent1': "A man is playing guitar on stage.",
            'sent2': "Someone is strumming a musical instrument in front of an audience."
        },
        {
            'name': 'Low Similarity (Unrelated)',
            'sent1': "The stock market crashed causing panic.",
            'sent2': "An octopus is swimming in the ocean."
        },
        {
            'name': 'Identical Sentences',
            'sent1': "The capital of France is Paris.",
            'sent2': "The capital of France is Paris."
        }
    ]

    print("\n2. Computing similarities for example pairs...\n")

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['name']}")
        print("-" * 70)
        print(f"Sentence 1: {example['sent1']}")
        print(f"Sentence 2: {example['sent2']}")

        # Encode sentences
        emb1 = model.encode([example['sent1']])[0]
        emb2 = model.encode([example['sent2']])[0]

        # Compute similarities
        xi = symmetric_xi(emb1, emb2)
        cosine = cosine_similarity_score(emb1, emb2)

        print(f"\nResults:")
        print(f"  Xi (symmetric):    {xi:.4f}")
        print(f"  Cosine similarity: {cosine:.4f}")
        print(f"  Difference:        {abs(xi - cosine):.4f}")

        # Interpretation
        if xi < cosine:
            print(f"  → Xi is more conservative (stricter rank correlation)")
        elif xi > cosine:
            print(f"  → Xi is more lenient (stronger rank correlation)")
        else:
            print(f"  → Xi and cosine agree")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("• Xi values are typically lower than cosine (conservative)")
    print("• Xi focuses on rank structure, cosine on magnitude alignment")
    print("• Both metrics agree on overall similarity patterns")
    print("• Xi provides a complementary perspective for ensemble methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
