from agents import MultiAgentRAG

system = MultiAgentRAG("knowledge.md")
result = system.run("Explain RAG and multi-agent systems")
assert "PlannerAgent" in result
assert "RetrieverAgent" in result
assert "WriterAgent" in result
print("Project 1 smoke test passed.")
