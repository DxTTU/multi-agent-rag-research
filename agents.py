from dataclasses import dataclass
from typing import Dict
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


@dataclass
class AgentResult:
    agent: str
    content: str
    metadata: Dict


class PlannerAgent:
    name = "PlannerAgent"

    def run(self, question: str) -> AgentResult:
        q = question.lower()
        tasks = ["identify the main concepts"]
        if "rag" in q or "retrieval" in q:
            tasks.append("retrieve evidence about retrieval and grounding")
        if "agent" in q:
            tasks.append("retrieve evidence about multi-agent orchestration")
        if "pytorch" in q or "tensorflow" in q:
            tasks.append("retrieve evidence about the requested ML frameworks")
        return AgentResult(self.name, "\n".join(f"- {t}" for t in tasks), {"tasks": tasks})


class TorchReranker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RetrieverAgent:
    name = "RetrieverAgent"

    def __init__(self, knowledge_path: str):
        self.chunks = [c.strip() for c in re.split(r"\n\s*\n", open(knowledge_path, encoding="utf-8").read()) if c.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks).toarray().astype("float32")
        torch.manual_seed(7)
        self.reranker = TorchReranker(self.matrix.shape[1])
        # Make the neural score an actual part of retrieval, while keeping lexical relevance dominant.
        with torch.no_grad():
            neural = self.reranker(torch.tensor(self.matrix)).numpy()
        self.neural_bias = (neural - neural.mean()) / (neural.std() + 1e-6)

    def run(self, question: str, top_k: int = 4) -> AgentResult:
        qv = self.vectorizer.transform([question]).toarray().astype("float32")[0]
        lexical = self.matrix @ qv
        scores = lexical + 0.05 * self.neural_bias
        idx = np.argsort(scores)[::-1][:top_k]
        passages = [{"rank": i + 1, "score": float(scores[j]), "text": self.chunks[j]} for i, j in enumerate(idx)]
        return AgentResult(self.name, "\n\n".join(f"[Source {x['rank']}] {x['text']}" for x in passages), {"passages": passages})


class TensorFlowIntentClassifier:
    def __init__(self):
        texts = ["explain rag", "what is retrieval", "multi agent system", "agent workflow", "pytorch model", "tensorflow model"]
        labels = np.array([0, 0, 1, 1, 2, 2])
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        if TF_AVAILABLE:
            self.backend = "TensorFlow/Keras"
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax")
            ])
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            self.model.fit(X, labels, epochs=20, verbose=0)
        else:
            self.backend = "NumPy fallback (TensorFlow unavailable in this runtime)"
            self.centroids = np.vstack([X[labels == k].mean(axis=0) for k in range(3)])

    def predict(self, question: str) -> int:
        X = self.vectorizer.transform([question]).toarray().astype("float32")
        if TF_AVAILABLE:
            return int(np.argmax(self.model.predict(X, verbose=0)[0]))
        return int(np.argmax(X @ self.centroids.T))


class AnalystAgent:
    name = "AnalystAgent"

    def run(self, question: str, retrieved: AgentResult, intent: int) -> AgentResult:
        return AgentResult(
            self.name,
            f"Intent class: {intent}\nGrounded evidence:\n{retrieved.content}\n\nGrounding rule: answer only from retrieved evidence.",
            {"intent": intent},
        )


class WriterAgent:
    name = "WriterAgent"

    def run(self, question: str, analysis: AgentResult) -> AgentResult:
        return AgentResult(
            self.name,
            f"Question: {question}\n\nThe agent team planned the task, retrieved local evidence, analyzed it, and prepared a grounded response.\n\n{analysis.content}",
            {},
        )


class MultiAgentRAG:
    def __init__(self, knowledge_path: str):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(knowledge_path)
        self.intent = TensorFlowIntentClassifier()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

    def run(self, question: str) -> str:
        p = self.planner.run(question)
        r = self.retriever.run(question)
        a = self.analyst.run(question, r, self.intent.predict(question))
        w = self.writer.run(question, a)
        return (
            f"=== {p.agent} ===\n{p.content}\n\n"
            f"=== {r.agent} ===\n{r.content}\n\n"
            f"=== {a.agent} ===\n{a.content}\n\n"
            f"=== {w.agent} ===\n{w.content}\n\n"
            f"TensorFlow backend: {self.intent.backend}"
        )
