# Project 1 — Multi-Agent RAG Research Assistant

## What it actually does
Takes a natural-language research question and sends it through four cooperating stages: Planner -> Retriever -> Analyst -> Writer.

- Retrieves relevant paragraphs from `knowledge.md` using TF-IDF.
- Uses a PyTorch neural scoring component as a reranking signal.
- Uses a TensorFlow/Keras intent classifier when TensorFlow is installed.
- If TensorFlow is unavailable, the project uses a deterministic NumPy fallback so the rest of the project remains runnable.
- Prints retrieved source passages so the response is grounded and traceable.

## Run
```
pip install -r requirements.txt
python main.py
```

Smoke test: `python test_project.py`
