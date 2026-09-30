"""
Live RAG quality gate (pytest -m eval). Runs against the STAGING Pinecone index with the
semantic cache bypassed, scoring the *actual* retrieved contexts with Ragas.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

FAITHFULNESS_MEAN_MIN = 0.95
FAITHFULNESS_CASE_MIN = 0.80
CONTEXT_PRECISION_MIN = 0.85
ANSWER_RELEVANCY_MIN = 0.85

GOLDEN = json.loads((Path(__file__).parent / "golden_dataset.json").read_text())

if os.environ.get("OPENAI_API_KEY", "").startswith("sk-test"):
    pytest.skip("live credentials not configured", allow_module_level=True)


@pytest.fixture(scope="module")
def pipeline():
    import main_pipeline
    return main_pipeline


def _user(group):
    return {"user_id": "ci-eval-bot", "clearance_group": group}


def test_unanswerable_questions_are_refused(pipeline):
    for case in GOLDEN["unanswerable"]:
        r = pipeline.execute_clinical_query(_user(case["clearance_group"]), case["question"], use_cache=False)
        assert not r.grounded, f"expected refusal for out-of-scope question: {case['question']!r}"


def test_rag_quality_benchmarks(pipeline):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

    from config import get_settings

    samples = []
    for case in GOLDEN["answerable"]:
        r = pipeline.execute_clinical_query(_user(case["clearance_group"]), case["question"], use_cache=False)
        assert r.grounded, f"pipeline refused an answerable question: {case['question']!r}"
        assert r.contexts, "grounded answer must carry retrieved contexts"
        samples.append({
            "user_input": case["question"],
            "response": r.answer,
            "retrieved_contexts": r.contexts,
            "reference": case["reference"],
        })

    s = get_settings()
    judge = LangchainLLMWrapper(ChatOpenAI(model=s.llm_model, temperature=0.0,
                                           api_key=s.openai_api_key.get_secret_value()))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=s.embedding_model,
                                                      api_key=s.openai_api_key.get_secret_value()))
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), ResponseRelevancy()]
    result = evaluate(dataset=EvaluationDataset.from_list(samples), metrics=metrics, llm=judge, embeddings=emb)
    df = result.to_pandas()

    faith, prec, rel = (df[m.name] for m in metrics)
    report = {
        "faithfulness_mean": faith.mean(), "faithfulness_min": faith.min(),
        "context_precision_mean": prec.mean(), "answer_relevancy_mean": rel.mean(),
    }
    print("\n====== RAG QUALITY REPORT ======")
    for k, v in report.items():
        print(f"{k:28s} {v:.4f}")

    assert not df[[m.name for m in metrics]].isna().any().any(), "a metric failed to compute (NaN)"
    assert report["faithfulness_mean"] >= FAITHFULNESS_MEAN_MIN, "hallucination risk: mean faithfulness too low"
    assert report["faithfulness_min"] >= FAITHFULNESS_CASE_MIN, "hallucination risk: a case scored very low"
    assert report["context_precision_mean"] >= CONTEXT_PRECISION_MIN, "retrieval precision regression"
    assert report["answer_relevancy_mean"] >= ANSWER_RELEVANCY_MIN, "answer relevancy regression"
