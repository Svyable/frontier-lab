# Orinth integration — next PR, not implemented

Upstream inspected: Onestep-AI-Labs/orinth main, October 8, 2026.

- backend/app/services/serving.py uses a managed llama.cpp server and only permits llm_gguf.
- backend/app/api/routers/serving.py provides GGUF lifecycle and SSE chat proxy.
- backend/app/ml/model_registry.py houses ModelSpec and registered predictors.
- backend/app/training/runners/llm_sft.py runs Transformers/PEFT on Apple MPS, not MLX-LM.
- backend/app/api/routes.py aggregates FastAPI domain routers.

Do not edit or replace existing GGUF serving when introducing MLX. Suggested boundaries:

1. **DecisionProvider HTTP adapter**: backend/app/services/decision_provider.py; POST /api/decisions router; type-aware schema checks; localhost-only teacher; mocked API tests.
2. **Model registry metadata**: distinguish Clef teacher provenance from generated student artifact, and capabilities generate / decide_pseudolabel / decide_joint_head.
3. **MLX process supervisor**: dedicated worker for local student inference, explicit model unload and readiness; no second model while training on 24 GB unified memory.
4. **Distillation job**: scenario-group-aware data manifests, retained teacher distributions, human-gold evaluation. Avoid losing source revision when exporting.
5. **Dedicated head research**: train a new student-compatible schema head on frozen 4B features; actual KL/Brier/calibration evaluation before any parity claim.

Run Orinth make check and new route tests on every PR. A real macOS Metal smoke test is necessary before claiming the inference and training commands work on the M4. LittleBit compression remains separate and its noncommercial code must not be silently imported.
