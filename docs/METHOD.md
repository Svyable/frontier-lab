# Method

## What we chase

Ideas in the neighborhood of:

- Extremely small active-parameter or single-core training regimes
- Block-routed / sparse MoE shaped for hardware efficiency
- Early probes: induction, positional shift, arithmetic, extractive QA
- Post-training sequences that compound rather than one-shot magic
- Failures invisible in the loss curve (expert collapse, junk-token argmax, routing stats, untrained vocab rows)
- Honest treatment of curated / generated corpora as distillation

## Selection bar

Ship or deepen an idea only if at least one holds:

1. It tests a **constraint** (core, memory, tokens, latency) others ignore.
2. It adds a **diagnostic** others can reuse.
3. It produces a **public artifact** (card, Space, dataset, writeup) worth starring.

## Experiment hygiene

- Freeze hardware / software pins in the experiment README.
- Separate train metrics from eval probes.
- Log negative results; they are first-class.
- Link every published Hub artifact back to a commit or tag here.

## Publishing

See `hub/PUBLISH.md`. Model cards must include license, intended use, limitations, training data provenance, and evaluation caveats.
