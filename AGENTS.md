# AGENTS.md

Rules for Frontier Lab and collaborators in this repository.

## Mission

Explore frontier ideas in efficient / small / hardware-aware model development and **publish the awesome stuff** to Hugging Face. Primary inspiration: extreme single-core / AMX-class training and early-emergent reasoning probes (see the AMX reasoning line of work on Hugging Face).

## Hard rules

- Public by default. Do not commit secrets, API keys, private data, or unlicensed corpora.
- Credit upstream papers, models, and datasets clearly on every writeup and model card.
- Never invent benchmarks, SOTA claims, or training curves. Use `TODO` / `UNVERIFIED` when evidence is missing.
- Prefer diagnostics that transfer (probes, ablations, histograms) over narrative that only celebrates wins.
- Escalate training spend, org-level HF publishes-as-Svyable, and license choices to Chief of Staff / the user.
- Hugging Face posts need an authenticated HF connector; if auth fails, pause and surface it — do not scrape around it.

## Workflow

1. Capture the idea (`ideas/`).
2. Design a minimal experiment (`experiments/`).
3. Record honest results (including failures).
4. Write a short public note (`writeups/`).
5. Ship to Hub per `hub/PUBLISH.md` (model / dataset / Space / paper page as appropriate).

## Tone

Curious, precise, slightly skeptical. Celebrate real signal; document the bugs.
