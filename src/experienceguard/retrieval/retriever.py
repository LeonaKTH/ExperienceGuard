"""Deterministic BM25 plus optional local, frozen ONNX embeddings. No downloads."""

from __future__ import annotations
from collections import Counter
import math
import re
from pathlib import Path
import numpy as np
from ..types import Candidate
from ..utils.core import token_upper, file_hash


def tokenize(text):
    return re.findall(r"[\w]+", text.casefold())


class LocalEmbedding:
    def __init__(self, path, expected_hash=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(path)
        model = path / "model.onnx"
        self.sha256 = file_hash(model)
        self.identity = {
            "model_sha256": self.sha256,
            "tokenizer_sha256": file_hash(path / "tokenizer.json"),
            "pooling": "attention-mask mean + L2",
            "truncation": 256,
            "execution_provider": "CPUExecutionProvider",
        }
        if expected_hash and expected_hash != self.sha256:
            raise ValueError("embedding model hash mismatch")
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        data = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            ),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        hidden = self.session.run(
            None, {i.name: data[i.name] for i in self.session.get_inputs()}
        )[0]
        mask = data["attention_mask"][:, :, None]
        vectors = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
        return vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
        )


class MemoryRetriever:
    def __init__(self, experiences, embedding=None, k1=1.5, b=0.75, renderer=None):
        if renderer is None:
            from ..memory.normalizer import ExperienceNormalizer

            renderer = ExperienceNormalizer()
        self.renderer = renderer
        self.experiences = sorted(experiences, key=lambda e: e.experience_id)
        if len({e.experience_id for e in self.experiences}) != len(self.experiences):
            raise ValueError("duplicate experience IDs")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b, self.embedding = k1, b, embedding
        self.docs = [
            tokenize(e.goal + " " + " ".join(c["tool"] for c in e.tool_calls))
            for e in self.experiences
        ]
        self.df = Counter(t for doc in self.docs for t in set(doc))
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.vectors = (
            embedding.encode([e.goal for e in self.experiences])
            if embedding and self.experiences
            else None
        )

    def retrieve(
        self, query, *, top_k=3, budget=2048, excluded_tasks=(), excluded_families=()
    ):
        if top_k < 0 or budget < 0:
            raise ValueError("invalid retrieval limits")
        tokens, n, scores = tokenize(query), len(self.docs), []
        sem = (
            self.vectors @ self.embedding.encode([query])[0]
            if self.vectors is not None
            else None
        )
        norm = self.renderer
        for index, (exp, doc) in enumerate(zip(self.experiences, self.docs)):
            if (
                exp.source_task in excluded_tasks
                or exp.source_family in excluded_families
            ):
                continue
            tf = Counter(doc)
            score = sum(
                math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                * tf[t]
                * (self.k1 + 1)
                / (
                    tf[t]
                    + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avg, 1))
                )
                for t in tokens
                if tf[t]
            )
            raw = token_upper(norm.render(exp, "raw"))
            normalized = token_upper(norm.render(exp))
            if min(raw, normalized) > budget:
                continue
            scores.append(
                Candidate(
                    exp,
                    score,
                    float(sem[index]) if sem is not None else None,
                    raw,
                    normalized,
                )
            )
        # All policies receive THIS pool. No per-policy recall expansion.
        return tuple(
            sorted(
                scores, key=lambda c: (-c.retrieval_score, c.experience.experience_id)
            )[:top_k]
        )
