# blg-py — BPMN Learn & Grade (Back-end)

REST API back-end for **BPMN Learn & Grade (BLG)**, a tool that automatically grades student BPMN process-model submissions against a reference rubric.

Consumed by the companion front-end: [blg-web](https://github.com/ETolboom/blg-web).

---

## Features

- **Automated grading** — runs a configurable set of structural, semantic and behavioral checks against student BPMN submissions.
- **Behavioral rule engine** — graph-based workflow rules (nodes + edges) capture expected BPMN sequences; evaluated with AND/XOR branch logic.
- **Semantic similarity** — uses `sentence-transformers/all-mpnet-base-v2` for label matching so minor wording differences don't break grading.
- **Formal control-flow analysis** — deadlock and dead-activity detection via a compiled Rust extension (`blg`).
- **Excel export** — per-submission and bulk grading results exportable as `.xlsx`.
- **Rule groups** — combine multiple behavioral rules under XOR (alternative solutions) or AND (all required) conditions.

---

## Architecture

```
blg-py/
├── main.py                        # FastAPI app entry point & lifespan setup
├── dependencies.py                # FastAPI dependency injection helpers
├── checks/
│   ├── __init__.py                # Check base class, CheckResult, CheckComplexity
│   ├── manager.py                 # CheckRegistry (auto-discovery) & CheckManager
│   └── implementations/          # Auto-loaded check plugins
│       ├── behavioral.py          # Behavioral rule traversal engine
│       ├── control_flow.py
│       ├── semantic.py
│       ├── task_coverage.py
│       ├── task_type.py
│       └── resources.py
├── routers/
│   ├── submissions.py             # /api/submissions
│   ├── rubric.py                  # /api/rubric
│   ├── checks.py                  # /api/checks
│   ├── behavioral_rules.py        # /api/behavioral-rules
│   └── behavioral_rule_groups.py  # /api/behavioral-rule-groups
├── bpmn/
│   ├── bpmn.py                    # BPMN XML parser & graph traversal
│   └── struct.py                  # Pool, Lane, PoolElement, FlowElement dataclasses
├── rules/
│   └── manager.py                 # BehavioralRuleManager (disk I/O for rules & groups)
├── rubric/
│   └── __init__.py                # Rubric, RubricCriterion, SubmissionResult models
├── services/
│   └── submissions.py             # SubmissionService (file management, export)
├── utils/
│   └── similarity.py              # Sentence-transformer similarity helpers
├── src/                           # Rust source for the blg extension (PyO3 / maturin)
└── example/                       # Sample data directory (rubric, submissions, rules)
    ├── rubric.json
    ├── reference.bpmn
    ├── supplement.pdf
    ├── submissions/
    ├── rules/
    └── templates/
```

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| [Rust toolchain](https://rustup.rs/) | Compiling the `blg` extension |
| [maturin](https://maturin.rs/) | Building the Rust/Python bridge |

---

## Installation

### 1. Install Python dependencies

```bash
pip install .
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_md
# uv users:
uv run -- spacy download en_core_web_md
```

### 3. Build the Rust extension

```bash
maturin develop
```

> Re-run `maturin develop` whenever you change any file under `src/`.

---

## Running

```bash
python main.py example
```

The server starts at `http://127.0.0.1:8000`. The `example/` directory is used as the data root (rubric, submissions, rules).

Swap `example` for any other directory that contains the expected layout.

### Bundling the front-end

The back-end will automatically serve the compiled [blg-web](https://github.com/ETolboom/blg-web) front-end if a `static/` directory exists at the project root. All API routes (`/api/*`) take priority; everything else falls back to `index.html` so SPA client-side routing works correctly.

```bash
# 1. Build the front-end (from the blg-web repository)
npm run build          # output lands in blg-web/dist/

# 2. Copy the build output into blg-py/static/
cp -r ../blg-web/dist/ ./static/

# 3. Start the server as normal — the UI is now served at http://127.0.0.1:8000
python main.py example
```

The `static/` directory is listed in `.gitignore` and is not checked in to this repository.

---

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/rubric` | Fetch the current rubric |
| `POST` | `/api/rubric` | Create a rubric via onboarding payload |
| `DELETE` | `/api/rubric/criteria/{id}` | Remove a rubric criterion |
| `GET` | `/api/submissions` | List all student submissions |
| `POST` | `/api/submissions` | Upload `.bpmn` files |
| `GET` | `/api/submissions/{filename}` | Download raw BPMN XML |
| `GET` | `/api/submissions/export` | Export a single result as `.xlsx` |
| `GET` | `/api/submissions/export/all` | Export all results as `.xlsx` |
| `GET` | `/api/checks` | List all registered checks |
| `POST` | `/api/checks/analyze` | Analyze a submission against the rubric |
| `POST` | `/api/checks/analyze/all` | List applicable checks for a given model |
| `GET` | `/api/behavioral-rules` | List behavioral rules |
| `POST` | `/api/behavioral-rules` | Create a rule |
| `PUT` | `/api/behavioral-rules/{id}` | Update a rule |
| `DELETE` | `/api/behavioral-rules/{id}` | Delete a rule |
| `POST` | `/api/behavioral-rules/{id}/validate` | Validate a rule against a BPMN model |
| `GET` | `/api/behavioral-rule-groups` | List rule groups |
| `POST` | `/api/behavioral-rule-groups` | Create a rule group |
| `PUT` | `/api/behavioral-rule-groups/{id}` | Update a rule group |
| `DELETE` | `/api/behavioral-rule-groups/{id}` | Delete a rule group |
| `POST` | `/api/behavioral-rule-groups/{id}/validate` | Validate a group against a BPMN model |

Interactive API docs are available at `http://127.0.0.1:8000/docs` when the server is running.

---

## Adding a New Check

1. Create a `.py` file in `checks/implementations/`.
2. Subclass `Check` and define the required `ClassVar` fields (`id`, `name`, `description`, `check_complexity`, `input_scheme`).
3. Implement `analyze()` and `is_applicable()`.

The check is auto-discovered and registered on the next server start — no wiring required.

---

## Linting & Formatting

```bash
ruff check .
ruff format .
```

Configured in `pyproject.toml` (rules: E, F, UP; double-quote style).

---

## Acknowledgements

The `src/` directory contains adapted code from the [`rust_bpmn_analyzer`](https://github.com/timKraeuter/rust_bpmn_analyzer) project by [Tim Kräuter](https://github.com/timKraeuter), used under the MIT licence. See `src/README.md` for details.
