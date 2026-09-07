# Frontier Lab

Public lab for **frontier-efficient** language model ideas — tiny active-parameter budgets, hardware-aware training (including Intel AMX / single-core regimes), sparse MoE, curricula that elicit early reasoning, compounding post-training, and diagnostics that catch failures the loss curve hides.

Inspired by work such as [gdiamos/amx-reasoning-v1-instruct](https://huggingface.co/gdiamos/amx-reasoning-v1-instruct) (*Outrageously Small Neural Networks: Emergent Basic Reasoning at 6,616 tok/sec on One Intel AMX Core*). This repo is independent exploration; it is not affiliated with that project’s authors.

**Agent:** Frontier Lab (Svyable)  
**Publish surface:** [Hugging Face Hub](https://huggingface.co/) — models, datasets, Spaces, paper pages.

## North stars

1. **Capability under harsh constraints** — what appears early when FLOPs, cores, or active parameters are tiny?
2. **Diagnostics over vibes** — prefer probes, ablations, and measurement traps over smooth loss curves.
3. **Data as distillation** — treat curated / LM-generated corpora honestly; credit upstream.
4. **Ship in public** — writeups and artifacts belong on GitHub + Hugging Face, with clear licenses and cards.

## Layout

```text
ideas/          Short notes on frontier concepts worth trying
experiments/    Run configs, logs pointers, result summaries
writeups/       Longer posts / paper-style notes
hub/            Manifests and checklists for HF publishes
AGENTS.md       Rules for the Frontier Lab agent
docs/METHOD.md  How we pick, run, and publish work
```

## Getting started

1. Drop an idea under `ideas/` (use `templates/idea.md`).
2. If it graduates, open an `experiments/<slug>/` folder.
3. When something is worth sharing, follow `hub/PUBLISH.md` and post to Hugging Face.
4. Never invent benchmark numbers; mark TODOs instead.

## Status

Scaffolding for Frontier Lab on [Svyable](https://github.com/Svyable). First explorations TBD.
