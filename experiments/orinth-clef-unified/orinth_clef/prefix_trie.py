"""Prefix-sharing candidate likelihood scorer for a frozen MLX transformer.

Compressed trie edges allow one model call for a shared token prefix.
Reference: candidate_sequences.score_sequences. Not calibrated probabilities.
"""
import copy
import math

from .decision_lens import select_from_logits
from .schema import SchemaError


def build_trie(candidates):
    """Pure-Python trie; leaf labels preserve the input's candidate identity."""
    root = {"children": {}, "labels": []}
    if not candidates:
        raise SchemaError("no candidate sequences")
    for label, ids in candidates.items():
        if not ids:
            raise SchemaError("empty candidate sequence")
        node = root
        for token in ids:
            if not isinstance(token, int) or token < 0:
                raise SchemaError("invalid candidate token ID")
            node = node["children"].setdefault(token, {"children": {}, "labels": []})
        node["labels"].append(label)
    return root


def trie_stats(root):
    """Count unique prefix nodes and terminal sequences, without MLX."""
    nodes, terminals = 0, 0
    pending = [root]
    while pending:
        node = pending.pop()
        nodes += 1
        terminals += len(node["labels"])
        pending.extend(node["children"].values())
    return {"unique_prefix_tokens": nodes - 1, "candidates": terminals}


def score_sequences_trie(model, cache, next_logits, candidates):
    """Return argmax + normalized heuristic scores using compressed trie edges.

    The shared input cache is never modified. A branch cache is copied at
    each trie fork. Token-sequence scores match the reference mean log-prob
    objective, subject to normal floating-point variation.
    """
    import mlx.core as mx

    root = build_trie(candidates)
    mx.eval(next_logits)
    scored = {}

    def visit(node, prefix_cache, prefix_logits, accumulated, depth):
        # A candidate may be a prefix of another candidate.
        for label in node["labels"]:
            scored[label] = accumulated / depth
        for token, child in node["children"].items():
            # Compress a nonterminal single-child chain into one model call.
            segment = [token]
            endpoint = child
            while not endpoint["labels"] and len(endpoint["children"]) == 1:
                next_token, endpoint = next(iter(endpoint["children"].items()))
                segment.append(next_token)

            # A terminal-only edge need not evaluate its last token; no
            # successor distribution is needed. Internal forks do.
            needs_next = bool(endpoint["children"])
            forward_tokens = segment if needs_next else segment[:-1]
            if forward_tokens:
                branch_cache = copy.deepcopy(prefix_cache)
                output = model(mx.array([forward_tokens]), cache=branch_cache)[0]
                mx.eval(output)
            else:
                branch_cache = prefix_cache
                output = None

            total = accumulated
            logits = prefix_logits
            for i, token_id in enumerate(segment):
                total += float((logits[token_id] - mx.logsumexp(logits)).item())
                if i + 1 < len(segment):
                    logits = output[i]
            if needs_next:
                visit(endpoint, branch_cache, output[-1], total, depth + len(segment))
            else:
                for label in endpoint["labels"]:
                    scored[label] = total / (depth + len(segment))

    visit(root, cache, next_logits, 0.0, 0)
    if set(scored) != set(candidates) or not all(math.isfinite(x) for x in scored.values()):
        raise SchemaError("invalid trie candidate scores")
    return select_from_logits(candidates, [scored[label] for label in candidates])
