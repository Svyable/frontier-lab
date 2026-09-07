# Frontier Lab

Public lab that **researches what’s next** — scanning Hugging Face and frontier papers — then ships **stupidly simple tricks** that 10x (or otherwise dramatically improve) model or approach performance.

AMX-class tiny MoE work such as [gdiamos/amx-reasoning-v1-instruct](https://huggingface.co/gdiamos/amx-reasoning-v1-instruct) is one inspiration, not a cage. Stay broad across recipes, data, routing, decoding, post-training, and inference.

**Agent:** Frontier Lab (Svyable)  
**Publish surface:** [Hugging Face Hub](https://huggingface.co/) — models, datasets, Spaces, paper pages.

## North stars

1. **Simple levers, large gains** — can you explain the trick in a paragraph and measure it cleanly?
2. **Self-directed research** — HF + papers decide the backlog when nobody named a task.
3. **Diagnostics over vibes** — probes, ablations, and measurement traps beat smooth loss curves.
4. **Ship in public** — writeups and artifacts on GitHub + Hugging Face, with clear licenses and cards.

## Layout

```text
ideas/          Short notes on tricks worth trying
experiments/    Run configs, logs pointers, result summaries
writeups/       Longer posts / paper-style notes
hub/            Manifests and checklists for HF publishes
AGENTS.md       Rules for the Frontier Lab agent
docs/METHOD.md  How we pick, run, and publish work
```

## Getting started

1. Research → drop an idea under `ideas/` (use `templates/idea.md`).
2. If it graduates, open an `experiments/<slug>/` folder.
3. When something is worth sharing, follow `hub/PUBLISH.md` and post to Hugging Face.
4. Never invent benchmark numbers; mark TODOs instead.

## Status

Active lab on [Svyable](https://github.com/Svyable). Charter: broad, simple, measured, public.
