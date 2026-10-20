from agents import MultiAgentRAG

if __name__ == "__main__":
    system = MultiAgentRAG("knowledge.md")
    question = input("Ask a research question: ").strip()
    if not question:
        question = "How does a multi-agent RAG system use retrieval, PyTorch and TensorFlow?"
    print("\n" + system.run(question))
