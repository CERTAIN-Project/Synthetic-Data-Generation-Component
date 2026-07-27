# Synthetic Data Generation Component Simulation Framework

**Synthetic Data Generation Component** (Collaborative Heterogeneous Online Reasoning and Understanding Simulation) is an agentic simulation framework for generating realistic public deliberation data.

The framework spawns a configurable population of AI agents — each with a distinct persona, archetype, and behavioral profile — and places them in a simulated online discussion forum. Agents post comments and reply to each other, producing rich threaded conversations that mirror the dynamics of real civic deliberation platforms.

Activity is governed by a **simple discrete loop**: on each time step, every agent is checked to see whether it should post or act based on straightforward per-actor step intervals. The result is a chronologically ordered dataset of posts and replies, suitable for research in computational social science, deliberation platform design, NLP dataset generation, and AI evaluation.

---

## Table of Contents

- [How It Works](#how-it-works)
  - [Discrete Event Loop](#discrete-event-loop)
  - [Posting & Replying](#posting--replying)
  - [Actor Archetypes](#actor-archetypes)
  - [Actor Behavioral Parameters](#actor-behavioral-parameters)
- [Mixed Dataset Generation](#mixed-dataset-generation)
  - [Participant Types](#participant-types)
  - [Clean Dataset](#clean-dataset)
  - [Dirty Dataset](#dirty-dataset)
  - [Data Corruption Strategies](#data-corruption-strategies)
  - [Output Schema](#output-schema)
  - [Storage & Upload](#storage--upload)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [CLI (Basic Simulation)](#cli-basic-simulation)
  - [Local Scenario UI](#local-scenario-ui)
  - [Notebook (Mixed Dataset Pipeline)](#notebook-mixed-dataset-pipeline)
  - [Configuration](#configuration)
- [Output Format](#output-format)
- [License](#license)

---

## How It Works

### Discrete Event Loop

The simulation runs as a **simple loop over time steps**. On each step, every actor is checked against a configured posting interval and action interval. When an interval matches the current step, the corresponding procedure runs. The loop continues until the simulation horizon `T` (in minutes) is reached.

### Posting & Replying

When an actor's POST event fires, it either:
- writes a **new top-level comment** on the discussion topic, or
- **replies** to a recent post already in the shared history

The choice is governed by a per-actor `p_reply` probability. All generated text is conditioned on the discussion topic, the actor's persona, and the full conversation history accumulated so far. Experts can optionally augment their posts with live web search results.

### Actor Archetypes

| Archetype | Behavior |
|---|---|
| `citizen_b2c` | Casual conversational tone, moderate activity, broad engagement |


### Actor Behavioral Parameters

Each actor is parameterized by the following values, configurable in `src/config.py`:

| Parameter | Description |
|---|---|
| `post_every_n_steps` | Run the POST procedure every N loop steps |
| `action_every_n_steps` | Run the ACTION procedure every N loop steps (`0` disables) |
| `p_reply` | Probability of replying to an existing post vs. writing a new one |
| `theta_action` | Voting selectivity threshold (higher = votes less often) |

---

## Mixed Dataset Generation

The notebook `notebooks/mixed_dataset_simulation.ipynb` implements a full pipeline for generating **paired clean/dirty datasets** of discussion comments. This is designed to produce realistic, research-grade NLP corpora with controlled data quality degradation.

### Participant Types

The simulation uses **23 actors** in total:

| Type | Count | Metadata | Behavior |
|---|---|---|---|
| Identified (regular) | 15 | Full demographic profile | Comments + replies |
| Identified (rude/negative) | 2 | Full profile + `_archetype` marker | Regular tone |
| Anonymous | 6 | None | Comments + replies |

**Identified actor metadata schema:**
```json
{
  "FirstName": "Elena",
  "LastName": "Ivanova",
  "Age": 45,
  "DateOfBirth": "08/07/1979",
  "Nationality": "Bulgaria"
}
```

The two "rude" identified actors (`Vasilis Kritikos` — critical, `Despina Skepsi` — skeptical) carry an internal `_archetype` field that influences their LLM prompt. This field is **automatically stripped** from all exported JSON output.

### Clean Dataset

The **clean dataset** contains only records from identified actors with complete, unmodified metadata. Anonymous actor posts are excluded entirely.

```json
{
  "TopicTitle": "Benefits of remote work",
  "TopicSummary": "Tell me why remote work is good or bad...",
  "Dataset": [
    {
      "Id": "6bb266ef-7491-4797-bfba-28aa6107fb83",
      "Text": "I think remote work is honestly a double-edged sword...",
      "Metadata": {
        "FirstName": "Elena",
        "LastName": "Ivanova",
        "Age": 45,
        "DateOfBirth": "08/07/1979",
        "Nationality": "Bulgaria"
      }
    }
  ]
}
```

### Dirty Dataset

The **dirty dataset** includes all actors (identified + anonymous) and applies a set of controlled data corruptions to simulate real-world data quality issues. Every record — regardless of corruption — maintains a consistent schema with all six metadata keys always present.

**Anonymous actor records** appear with all metadata fields set to `null`:
```json
{
  "Id": "fc61be9f-9382-4bff-bdff-201b12d6092e",
  "Text": "I think what's getting lost...",
  "Metadata": {
    "FirstName": null,
    "LastName": null,
    "Age": null,
    "DateOfBirth": null,
    "Nationality": null
  }
}
```

### Data Corruption Strategies

Corruptions are applied stochastically per record during dirty dataset construction:

| Strategy | Probability | Effect |
|---|---|---|
| **Metadata shuffle** | 30% | Metadata is swapped with another randomly chosen identified actor's profile |
| **Field nullification** | 50% | 1–2 metadata fields are set to `null` (keys remain present) |
| **Value obfuscation** | 30% | `FirstName` replaced with `"***"`, or `Age` shifted by ±10 years |
| **Text truncation** | 40% | Comment text is cut short and appended with `"..."` |

These strategies can co-occur on the same record. Public metadata keys (`FirstName`, `LastName`, `Age`, `DateOfBirth`, `Nationality`) are present in the output; internal generation and evaluation labels are kept in sidecar annotations.

### Output Schema

Both clean and dirty datasets share the same top-level JSON structure:

```json
{
  "TopicTitle": "string",
  "TopicSummary": "string",
  "Dataset": [ ...records ]
}
```

Each record in `Dataset`:

| Field | Type | Notes |
|---|---|---|
| `Id` | `string` (UUID) | Unique identifier for the post |
| `Text` | `string` | Full or truncated comment text |
| `Metadata` | `object` | Always present; values may be `null` in dirty version |
| `Metadata.FirstName` | `string \| null` | — |
| `Metadata.LastName` | `string \| null` | — |
| `Metadata.Age` | `integer \| null` | May be shifted ±10 in dirty version |
| `Metadata.DateOfBirth` | `string \| null` | Format: `DD/MM/YYYY` |
| `Metadata.Nationality` | `string \| null` | — |

Internal generation and evaluation labels such as `Background`, `Opinion`, `AnswerQuality`, `BiasGroup`, and `WrongAnswerPattern` are not exported inside the generated clean/dirty datasets. They are written to `session/scenario_annotations_batch_N.json` so clustering and MLOps metrics can still use them without exposing them in the public dataset.

### Storage & Upload

Each simulation run produces an auto-numbered batch. Files are saved both **locally** and **uploaded to MinIO object storage**:

| File | Location |
|---|---|
| `session/simulation_output_clean_N.json` | Local filesystem |
| `session/simulation_output_dirty_N.json` | Local filesystem |
| `session/scenario_annotations_batch_N.json` | Local filesystem, internal analysis labels |
| `certain-results/simulation_output_clean_N.json` | MinIO bucket |
| `certain-results/simulation_output_dirty_N.json` | MinIO bucket |

The batch number `N` is automatically determined by scanning existing files in `session/` to avoid overwrites.

---

## Project Structure

```
Synthetic Data Generation Component-simulation-framework/
├── src/
│   ├── __main__.py             # CLI entry point
│   ├── simulation.py           # Core async event loop
│   ├── actors.py               # Actor behavior logic
│   ├── models.py               # Data models (ActorConfig, SharedHistory, etc.)
│   ├── config.py               # Global constants and actor parameter table
│   ├── llm_client.py           # OpenAI API client wrapper
│   ├── minio_manager.py        # MinIO client wrapper
│   ├── formatting.py           # Output formatting utilities
│   ├── temporal.py             # Event timing utilities
│   ├── procedures.py           # High-level simulation procedures
│   └── prompts/
│       ├── anthropic_system_prompts.py   # Per-archetype system prompts
│       └── anthropic_user_prompts.py     # Dynamic user prompt builders
├── notebooks/
│   ├── mixed_dataset_simulation.ipynb   # Mixed dataset generation pipeline
│   └── run_simulation.ipynb             # Basic simulation notebook
├── session/
│   └── simulation_output*.json          # Generated output files
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Setup

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/)
- A MinIO instance (for remote storage; optional for local-only use)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=

# MinIO (optional — required only for remote upload)
MINIO_ENDPOINT=your-minio-host:port
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_SECURE=true
```

---

## Usage

### CLI (Basic Simulation)

Runs the default simulation defined in `src/config.py` and writes output to `session/simulation_output.json`:

```bash
python -m src
```

### Local Scenario UI

Use the local UI when you want to create a custom clean/dirty bias test without editing Python files. The UI lets you define a batch, generate paired datasets, run clustering, inspect scatterplots, and read plain-language verdicts for the clean and dirty sets.

Start the UI from the `certain-sd/certain-synthetic-data` project folder. If you are in the parent repository folder, move into the app folder first:

```powershell
cd C:\Users\Administrator\Documents\GitHub\certain\certain-sd\certain-synthetic-data
```

Then start the UI:

```bash
python -m src.scenario_ui
```

Open:

```text
http://127.0.0.1:8769
```

If that port is already in use, choose another one:

```powershell
python -m src.scenario_ui --host 127.0.0.1 --port 8769
```

Then open `http://127.0.0.1:8769`.

#### Create Your Own Test

1. In the `Scenario builder` tab, set a new `Batch id`.
2. Keep `Overwrite existing` as `Yes` only if you intentionally want to replace an existing batch.
3. Choose `Run clustering = Yes` so the UI generates scatterplots, cluster assignments, summaries, and verdicts after dataset generation.
4. Choose `Text generation`:
   - `Fast fallback` for quick local tests without calling the LLM.
   - `LLM natural text` for realistic comments using the configured OpenAI credentials in `.env`.
5. Set `Total clean actors` and `Total dirty actors` for the dataset size.
6. Select the countries of origin and age buckets you want to include.
7. Choose a `Scenario template` or manually configure the deviation controls.
8. Click `Apply template` if using a template, or click `Build actor groups` for manual settings.
9. Review `Design Validation` to confirm actor totals and expected TVD signals.
10. Click `Generate datasets`.

The UI writes paired clean/dirty files under `session/` and clustering artifacts under `session/cluster_bias_outputs/batch_N/`.

#### Bias Controls

Use `Deviation grade` to control how far the dirty dataset diverges from the clean baseline:

- `1 - subtle`: small count changes and light opinion skew.
- `3 - medium`: visible targeted skew for scatterplot inspection.
- `5 - extreme`: strong over/under-representation, sharper opinion skew, and more partial/incorrect answers in biased groups.

The scenario designer also supports:

- `Total clean actors` / `Total dirty actors`: set dataset size directly.
- `Allocation strategy`: choose equal, population-like, selected-group skew, or custom allocation.
- `Deviation target`: stress demographics, opinions, answer quality, country-age interactions, hidden quality bias, or all signals.
- `Wrong-answer pattern`: choose how incorrect answers fail, such as off-topic, overgeneralized, stereotyped, missing safeguards, contradictory, or low-detail.
- `Scenario template`: quickly apply common designs such as mild demographic bias, strong age polarization, origin underrepresentation, answer-quality degradation, or hidden quality bias.
- `Mode mix`: split dirty data across balanced, demographic skew, opinion skew, and quality-stress records.
- `Design Validation`: preview expected actor totals and approximate TVD signals before generation.

Compound bias modes are topic-agnostic and operate on the selected origins and age buckets:

- `Balanced`: clean and dirty stay aligned.
- `Age polarization`: younger actors are overrepresented toward support, older actors toward opposition.
- `Origin pair polarization`: the first selected origin is skewed toward younger/supportive records, while the last selected origin is skewed toward older/opposing records.
- `First origin overrepresented`: one selected origin dominates the dirty dataset.
- `First origin underrepresented`: one selected origin is suppressed in the dirty dataset.
- `Answer-quality stress`: counts stay closer, but biased groups receive more partial/incorrect answers.

#### Inspect Results

After generation, open the `Batch data` tab.

Use the `Batch` selector to choose the batch you want to inspect. The tab shows:

- `Clean verdict`: a plain-language verdict for whether the clean set itself appears biased.
- `Dirty verdict`: a separate verdict for whether the dirty set appears biased.
- `Clean vs dirty verdict`: a comparative verdict explaining the strongest shift between the two sets, for example that a specific opinion is overrepresented in a specific age bucket.
- Summary metrics such as nationality TVD, opinion TVD, answer-quality TVD, and bias-group TVD.
- A central scatterplot panel with internal tabs:
  - `Clean vs dirty`
  - `Clean`
  - `Dirty`
- `Cluster Sets`: cluster-level distributions by dataset, cluster id, nationality, age, opinion, quality, and bias group.
- `Batch Data`: table views for clean records, dirty records, clean assignments, and dirty assignments, with a text filter.

The verdict text is intentionally written in simple language. Example:

```text
The dirty set appears strongly biased: in ages 18-29, the opinion "support" dominates compared with the rest of the dirty set.
```

#### Generated Files

For batch `N`, the UI creates:

| File | Purpose |
|---|---|
| `session/simulation_output_clean_N.json` | Public clean dataset |
| `session/simulation_output_dirty_N.json` | Public dirty dataset |
| `session/scenario_annotations_batch_N.json` | Internal labels for clustering and bias analysis |
| `session/scenario_config_batch_N.json` | Scenario configuration used to generate the batch |
| `session/cluster_bias_outputs/batch_N/*_scatterplot.html` | Interactive scatterplot HTML artifacts |
| `session/cluster_bias_outputs/batch_N/*_scatterplot.svg` | Static scatterplot SVG artifacts |
| `session/cluster_bias_outputs/batch_N/*_cluster_assignments.csv` | Per-record cluster assignments and scatterplot coordinates |
| `session/cluster_bias_outputs/batch_N/*_cluster_summary.json` | Cluster-level summaries |
| `session/cluster_bias_outputs/batch_N/*_bias_snapshot.json` | Dataset-level bias signals |
| `session/cluster_bias_outputs/batch_N/batch_N_clean_dirty_comparison.json` | Clean-vs-dirty comparison metrics |

### Notebook (Mixed Dataset Pipeline)

Open `notebooks/mixed_dataset_simulation.ipynb` and execute cells in order:

| Cell | Purpose |
|---|---|
| **1 — Imports** | Load dependencies and configure logging |
| **2 — Generate Participants** | Create 15 regular + 2 rude identified actors and 6 anonymous actors with synthetic metadata |
| **3 — Configuration** | Set topic, simulation horizon `T`, and actor counts |
| **4 — Create Actors** | Build `ActorConfig` objects (voting disabled for all actors) |
| **5 — Run Simulation** | Execute async simulation; produces `SharedHistory` object `H` |
| **6 — Analyze Results** | Display post counts per actor, verify no vote events |
| **7 — Export & Save** | Apply corruption strategies, create clean/dirty JSON, save locally and upload to MinIO |
| **8 — Verify Output** | Parse saved files and display sample records |

**Quick test vs. production:**

```python
T = 2.0    # ~40 comments — quick verification run
T = 180.0  # ~500 comments — full production dataset
```

### Configuration

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt-4.1-mini` | OpenAI model used for all LLM calls |
| `SIMULATION_HORIZON_MINUTES` | `0.6` | Simulated time horizon in minutes |
| `M_CANDIDATES` | `3` | Max posts an agent considers per ACTION event |
| `MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `TOPIC_TITLE` | *(configurable)* | Title of the deliberation topic |
| `TOPIC_SUMMARY` | *(configurable)* | Detailed topic description injected into all prompts |

---

## Output Format

### Basic Simulation (`python -m src`)

Writes a JSON array of chronologically ordered events to `session/simulation_output.json`:

```json
[
  {
    "id": "uuid",
    "t": 3.142,
    "type": "post",
    "actor": "Casual User 1",
    "archetype": "citizen_b2c",
    "text": "...",
    "reply_to": null
  },
  {
    "id": "uuid",
    "t": 4.017,
    "type": "vote",
    "actor": "Casual User 2",
    "archetype": "citizen_b2c",
    "target_id": "post-uuid",
    "vote": "upvote"
  }
]
```

### Mixed Dataset Simulation (Notebook)

Produces two paired files per batch — see [Output Schema](#output-schema) above.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
