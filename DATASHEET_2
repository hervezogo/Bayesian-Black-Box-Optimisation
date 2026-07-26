# Datasheet for the Black-Box Optimisation Challenge Dataset

> **Project:** Bayesian Black-Box Optimisation Capstone  
> **Programme:** Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence  
> **Documentation framework:** Adapted from *Datasheets for Datasets* (Gebru et al., 2021)  
> **Document status:** Repository-ready snapshot based on the supplied cumulative HTML notebook exports  
> **Primary objective:** Maximise eight unknown objective functions under a restricted evaluation budget

---

## 1. Introduction

### 1.1 Purpose

This datasheet documents the dataset created and extended during the Black-Box Optimisation (BBO) capstone challenge. The challenge contains eight independent objective functions, labelled **F1–F8**, whose analytical forms, gradients and internal mechanisms are not available to the participant. Each function can only be investigated by submitting a valid input vector and observing the scalar response returned by the hidden evaluator.

The dataset is therefore not a conventional collection of passively observed examples. It is the **complete experimental record of a sequential decision process**. Every student-generated observation was selected after analysing the data available at that point in time. Later observations are consequently conditioned on earlier ones, and their order is part of the dataset's meaning.

This distinction is important:

- a conventional supervised-learning dataset aims to represent an underlying population;
- this dataset aims to find high-performing regions of unknown functions as efficiently as possible;
- concentration around promising regions is therefore expected and intentional;
- the dataset records both what was observed and how the search evolved.

The purpose of this document is to explain the dataset's provenance, structure, statistical properties, quality controls, limitations and appropriate use. It does **not** attempt to reproduce the full modelling discussion. Gaussian Process specifications, acquisition-function choices, hyperparameter decisions and optimisation diagnostics should be documented separately in `MODEL_CARD.md` and the function notebooks.

### 1.2 Dataset identity

The dataset comprises eight related but independent tables. A row represents one evaluation:

\[
(\mathbf{x}, y), \qquad \mathbf{x} \in [0,1]^d,\quad y \in \mathbb{R}
\]

where:

- `x1, ..., xd` are continuous input coordinates;
- `d` is the dimensionality of the relevant function;
- `y` is the scalar response returned by the hidden function;
- the optimisation direction is **maximisation** for all eight functions.

For functions with negative responses, a larger value still represents a better result. For example, `-0.2` is preferable to `-1.0`.

### 1.3 Scope of this snapshot

The statistics in this datasheet are calculated from the latest cumulative tables visible in the supplied HTML exports. The exports are not all frozen at the same notebook checkpoint:

- F1 and F2 contain 12 appended query results and headings through Week 13;
- F3, F4, F6, F7 and F8 contain 7 appended query results and headings through Week 8;
- F5 contains 8 appended query results and headings through Week 10.

The resulting snapshot contains **242 observations**: 175 course-provided initial observations and 67 unique appended observations. If the repository contains later responses not present in these exports, the summary table should be regenerated before the document is treated as the definitive final release.

This explicit scope statement avoids a common documentation error: presenting statistics from mixed notebook versions as though they represented one synchronised final round.

---

## 2. Dataset at a Glance

### 2.1 Summary

| Property | Value |
|---|---|
| Number of objective functions | 8 |
| Function dimensionality | 2 to 8 input dimensions |
| Optimisation direction | Maximise `y` for every function |
| Input domain | Unit hypercube, `[0,1]^d` |
| Initial observations | 175 |
| Appended observations visible in supplied exports | 67 |
| Total observations in documented snapshot | 242 |
| Missing values in cumulative snapshot | 0 |
| Duplicate input vectors in cumulative snapshot | 0 |
| Input-bound violations in cumulative snapshot | 0 |
| Primary initial-data format | NumPy input and output arrays |
| Primary experimental record | Per-function notebooks and cumulative tables |
| Data sensitivity | Synthetic; no personal or confidential information |

### 2.2 Function-level snapshot

The values below are descriptive statistics of the supplied cumulative tables. **Best observed** means the largest response recorded in the snapshot; it is not a claim that the global optimum has been proven.

| Function | Dim. | Initial rows | Snapshot rows | Observed `y` range | Best observed input | Best observed `y` |
|---|---:|---:|---:|---:|---|---:|
| F1 | 2 | 10 | 22 | `[-0.003606063, 7.829289e-16]` | `[0.731010, 0.732975]` | `7.829289e-16` |
| F2 | 2 | 10 | 22 | `[-0.065624, 0.632404]` | `[0.714286, 0.025000]` | `0.632404` |
| F3 | 3 | 15 | 22 | `[-0.398926, -0.007682]` | `[0.434343, 0.525253, 0.444444]` | `-0.007682` |
| F4 | 4 | 30 | 37 | `[-32.625660, 0.255738]` | `[0.418919, 0.418919, 0.378378, 0.445946]` | `0.255738` |
| F5 | 4 | 20 | 28 | `[0.112940, 8662.482500]` | `[1.000000, 1.000000, 1.000000, 1.000000]` | `8662.482500` |
| F6 | 5 | 20 | 27 | `[-2.571170, -0.219614]` | `[0.446501, 0.395649, 0.641075, 0.823617, 0.197425]` | `-0.219614` |
| F7 | 6 | 30 | 37 | `[0.002701, 2.736811]` | `[0.001903, 0.183720, 0.418630, 0.234987, 0.305263, 0.623630]` | `2.736811` |
| F8 | 8 | 40 | 47 | `[5.592193, 9.940738]` | `[0.114426, 0.134779, 0.113350, 0.383082, 0.851282, 0.492447, 0.168010, 0.619063]` | `9.940738` |

### 2.3 Pedagogical function framing

The notebooks attach illustrative scenarios to the functions. These scenarios help explain why expensive black-box optimisation is useful, but they should be treated as **educational framing**, not as evidence that the observations came from operational radiation, pharmaceutical, warehouse or industrial systems.

| Function | Notebook framing | Data interpretation |
|---|---|---|
| F1 | Two-dimensional contamination or radiation-source localisation | Sparse 2D response with values concentrated extremely close to zero |
| F2 | Noisy two-parameter ML log-likelihood | 2D objective with a clearer positive high-value region |
| F3 | Three-compound side-effect optimisation | Negative transformed objective; values nearer zero are preferable |
| F4 | Four-parameter warehouse or surrogate-model tuning | Large negative tail with a small positive region |
| F5 | Four-input chemical-process yield | Strong scale growth towards a boundary corner |
| F6 | Higher-dimensional tuning problem | Five-dimensional negative objective; maximise by moving towards zero |
| F7 | Six-hyperparameter ML optimisation | Positive, high-dimensional objective with strong improvement near a boundary |
| F8 | Generic eight-dimensional optimisation / ML tuning | Highest-dimensional function with responses in a comparatively narrow numerical band |

---

## 3. Dataset Construction

### 3.1 Provenance and responsibility

The observations have two distinct sources:

1. **Course-provided warm-start observations**  
   Imperial College supplied initial input arrays and corresponding output arrays for each function. These observations established the starting design and were not selected by the student.

2. **Student-generated sequential observations**  
   Subsequent query points were selected using the accumulated dataset, surrogate-model outputs, acquisition functions, numerical diagnostics and function-specific reasoning. The hidden evaluator returned one scalar response for each submitted point.

This distinction should be preserved in any released version. At minimum, each row should carry a provenance field such as:

```text
source = "course_initial" | "student_query"
```

Where available, the round number and the rationale for the query should also be recorded. Without provenance, a downstream user cannot distinguish the fixed starting design from the adaptive observations generated by the optimisation policy.

### 3.2 Experimental lifecycle

The dataset grows through a repeated closed-loop process:

```mermaid
flowchart LR
    A[Course-provided initial observations] --> B[Cumulative function dataset]
    B --> C[Exploratory analysis and surrogate fitting]
    C --> D[Generate and compare candidate points]
    D --> E[Select one query for the function]
    E --> F[Submit query to hidden evaluator]
    F --> G[Receive scalar response]
    G --> H[Validate and append the new row]
    H --> B
```

The important feature is the feedback loop. The query-generation distribution is not fixed in advance: it changes when new evidence changes the model's view of the search space.

### 3.3 Collection strategy

The notebooks show a progression from broad search towards increasingly local refinement. Across functions, the collection workflow used combinations of:

- geometric inspection and grid-based exploration in low dimensions;
- Gaussian Process surrogate models;
- Upper Confidence Bound (UCB) for uncertainty-aware exploration;
- Expected Improvement (EI) for improvement-focused sampling;
- Probability of Improvement (PI) for tight local refinement;
- regular grids, random candidates and Sobol candidate sets;
- local trust regions around the current best observation;
- checks to avoid appending an already recorded `(x, y)` pair.

These modelling choices explain how the added rows were generated, but they do not alter the meaning of the raw dataset. The stored observation remains the submitted input and returned response, regardless of which acquisition rule proposed it.

### 3.4 Evolution of the sampling distribution

The dataset changes character over time:

- **Initial phase:** the course-provided design offers broad but sparse coverage.
- **Exploration phase:** queries probe uncertain, distant or boundary regions.
- **Localisation phase:** repeated evidence identifies regions likely to contain higher values.
- **Refinement phase:** later queries become more concentrated around current incumbents, ridges, corners or narrow response structures.

This evolution is visible in several functions:

- F1 moves from global 2D grid proposals to a very tight local search around approximately `[0.731, 0.733]`.
- F2 increasingly examines a region near the lower boundary of `x2`.
- F4 refines a compact interior region around values close to `0.4` in each dimension.
- F5 follows a strong trajectory towards the all-ones corner.
- F6 and F7 move from global higher-dimensional candidates to local candidate regions around improving incumbents.
- F8 concentrates later queries around a high-performing region characterised by low values in the first three coordinates and high `x5`.

These are properties of the collected data, not proof that the hidden functions have only one optimum.

### 3.5 Data lineage

The notebooks load initial data from paths of the form:

```text
Initial Data/function_k/initial_inputs.npy
Initial Data/function_k/initial_outputs.npy
```

The weekly workflow then reconstructs a cumulative table with columns:

```text
x1, x2, ..., xd, y
```

New points are represented in notebook code as `x_new` and `y_new`, validated, appended and displayed in the cumulative table.

For a durable repository release, a stronger lineage design would preserve three layers:

```text
data/
├── raw/
│   ├── function_1_initial_inputs.npy
│   ├── function_1_initial_outputs.npy
│   └── ...
├── queries/
│   ├── function_1_queries.csv
│   └── ...
└── processed/
    ├── function_1_cumulative.csv
    └── ...
```

The raw layer should be immutable. Query logs should be append-only. Processed cumulative files may be regenerated from the two upstream sources.

### 3.6 Recommended row schema

A release-quality tabular representation should contain more than coordinates and output:

| Field | Type | Description |
|---|---|---|
| `function_id` | string | Function label, `F1` to `F8` |
| `observation_id` | integer/string | Stable unique row identifier |
| `source` | categorical | `course_initial` or `student_query` |
| `round` | integer/null | Query round; null for initial rows if unavailable |
| `x1 ... xd` | float | Input coordinates |
| `y` | float | Returned objective value |
| `is_incumbent` | boolean | Whether the row established a new best value at collection time |
| `query_method` | string/null | UCB, EI, PI, manual probe, grid search, etc. |
| `query_rationale` | text/null | Short human-readable reason for selecting the point |
| `notebook_version` | string/null | Source notebook or commit identifier |

Only `x1 ... xd` and `y` are required to fit a surrogate. The additional fields are required to audit, reproduce and interpret the sequential experiment.

---

## 4. Dataset Characteristics

### 4.1 Data types and domains

Inputs and outputs are stored as numerical floating-point values. Input dimensionality varies by function, but each input coordinate in the supplied cumulative snapshot lies within `[0,1]`.

The objective scales are not comparable across functions. For example:

- F1 is concentrated around zero and includes values written in scientific notation;
- F4 ranges from a large negative value to a small positive value;
- F5 ranges from approximately `0.11` to more than `8,600`;
- F8 lies between approximately `5.59` and `9.94`.

A model or evaluation metric that pools raw outputs across functions would therefore be misleading. Any cross-function analysis must use function-specific normalisation or rank-based measures.

### 4.2 Function-specific statistical behaviour

#### F1 — sparse and numerically delicate

F1's recorded responses are dominated by extremely small magnitudes, alongside a negative observation near `-0.0036`. Many positive values are so close to zero that ordinary decimal formatting hides meaningful differences.

Implications:

- output transformations or log-scale visualisation may be needed for modelling diagnostics;
- numerical precision must be preserved;
- a model can appear well fitted while still failing to resolve a narrow response region;
- “near zero” should not be confused with missing data.

#### F2 — clearer low-dimensional structure

F2 has a wider and more interpretable response range than F1. The best supplied observation occurs near `x2 = 0.025`, and later queries inspect nearby boundary and interior points.

Implications:

- the 2D domain supports direct visual diagnostics;
- boundary behaviour matters;
- local refinement can be assessed visually, but the observed best remains only an incumbent.

#### F3 — negative transformed objective

All recorded F3 outputs are negative. Maximisation therefore means moving towards zero. The supplied snapshot improves from an initial best around the low negative range to `-0.007682`.

Implications:

- reporting “smaller absolute side effects” is clearer than calling the value negative performance;
- metrics and plots must preserve the maximisation direction;
- raw values should not be sign-flipped in the stored dataset merely to suit a software library.

#### F4 — asymmetric range and heavy negative tail

F4 has the widest negative range in the supplied snapshot: approximately `-32.63` to `0.256`. A small number of very poor observations can dominate mean, variance and kernel fitting.

Implications:

- robust descriptive statistics are more informative than the mean alone;
- outliers should be checked, not automatically removed;
- output scaling may improve model conditioning but must remain model-side preprocessing.

#### F5 — extreme scale growth and boundary optimum in the snapshot

F5 spans several orders of magnitude. Successive observations move towards the upper corner, with the largest value at `[1,1,1,1]`. Several distinct near-corner inputs return the same recorded output of approximately `7786.37135`.

Implications:

- the all-ones point is the best **observed** point, not necessarily a mathematically proven global optimum;
- identical outputs at distinct points are valid responses and must not be mistaken for duplicate rows;
- logarithmic visualisation can clarify the full range;
- boundary sampling is integral to the observed trajectory.

#### F6 — five-dimensional negative objective

F6 remains negative throughout the supplied snapshot, with improvement towards `-0.219614`. Higher dimensionality means that 27 observations provide extremely sparse coverage of the unit hypercube.

Implications:

- local improvement does not imply global coverage;
- uncertainty estimates can be sensitive to kernel assumptions;
- apparent coordinate importance should be treated as provisional.

#### F7 — high-dimensional improvement near a boundary

F7 reaches `2.736811` at an input with `x1` close to zero. Later nearby observations remain high, suggesting a locally promising region.

Implications:

- boundary-aware candidate generation is important;
- high local performance is credible evidence of a promising region, not proof that distant regions are inferior;
- 37 observations remain sparse in six dimensions.

#### F8 — eight-dimensional, relatively compressed output range

F8's responses occupy a narrower numerical range than F5 or F4, but its eight-dimensional domain is much larger. Later observations concentrate around a region that produces values close to `9.94`.

Implications:

- small response differences may matter;
- visualisation requires projections, slices or distance-based summaries;
- density in any 2D projection can create a false impression of adequate 8D coverage.

### 4.3 Data quality checks

The cumulative tables extracted from the supplied HTML exports were checked for basic integrity.

| Check | Result |
|---|---|
| Missing input or output values | None detected |
| Exact duplicate input vectors within a function | None detected |
| Exact duplicate complete rows | None detected |
| Inputs below 0 | None detected |
| Inputs above 1 | None detected |
| Scalar output present for each row | Yes |
| Consistent number of coordinates within each function | Yes |

These checks establish structural integrity, not scientific validity. They cannot determine whether the hidden evaluator returned an incorrect value, whether a query was submitted under the intended conditions, or whether all notebook versions are synchronised.

### 4.4 Data-preservation policy

Raw responses should be stored exactly as returned, including:

- negative values;
- values extremely close to zero;
- repeated output values at different inputs;
- poor exploratory observations;
- boundary evaluations;
- observations that later appear unhelpful.

Removing “bad” outcomes would introduce survivor bias and destroy the experimental history. If a response is suspected to be erroneous, the correct procedure is to flag it with metadata and retain the original value, not silently overwrite or delete it.

### 4.5 Preprocessing boundary

A clear separation should be maintained between **stored data** and **model-side transformations**.

#### Stored data

- original coordinates;
- original returned `y`;
- provenance and round metadata;
- no permanent rescaling or sign reversal.

#### Model-side processing

Depending on the function, notebooks use or imply:

- output normalisation through model settings;
- kernels with different length scales;
- numerical floors for predictive standard deviations;
- local candidate bounds;
- grid or Sobol candidate generation;
- temporary transforms or scaling for stability.

These operations may be necessary for modelling, but they should be reproducible from the raw data and should never replace it.

### 4.6 Biases

#### Adaptive sampling bias

Adaptive sampling is the defining design characteristic of the dataset. Queries are intentionally selected because they appear informative or promising. The final point cloud therefore over-represents regions favoured by the optimisation policy and under-represents regions judged unlikely to improve the objective.

This is not a defect for optimisation research. It becomes a problem only when the dataset is repurposed as though it were a representative sample of the whole domain.

#### Temporal dependence

Rows are not exchangeable. A later query could not have been selected in the same way without the earlier responses that shaped the model and the human decision process.

Randomly shuffling rows may be acceptable for fitting a static surrogate, but it removes the information required to reproduce sequential performance. Round order should therefore be preserved.

#### Incumbent and exploitation bias

Once a promising region is identified, subsequent queries cluster around it. This increases local resolution but reduces information about unexplored regions. The effect is particularly visible in F1, F4, F5, F7 and F8.

#### Dimensionality bias

The evaluation budget does not grow proportionally with the volume of the search space. High-dimensional functions therefore have much lower effective coverage than 2D functions, even when their row counts are larger.

#### Boundary bias

Several strong observations occur near domain boundaries or corners. Candidate-generation methods that clip to `[0,1]` can also increase the frequency of boundary points. This must be distinguished from genuine evidence that the hidden optimum lies on a boundary.

#### Policy bias

The dataset reflects the modelling choices, candidate sets, random seeds, trust-region sizes and human judgements used during collection. A different optimiser starting from the same initial observations would produce a different dataset.

### 4.7 Limitations

1. **Small sample size**  
   Each function contains only 22–47 observations in the supplied snapshot. This is appropriate for an expensive-query challenge but limits statistical certainty.

2. **Unknown ground truth**  
   The analytical functions and true global optima are not available. Regret cannot be measured exactly unless the course provides reference optima.

3. **Mixed notebook checkpoints**  
   The supplied HTML exports are not synchronised to the same final week. Snapshot statistics must not be described as one common final-round result without updating the source files.

4. **Sparse high-dimensional coverage**  
   F6–F8 contain too few observations to map their domains comprehensively.

5. **Limited metadata in the cumulative tables**  
   The tables contain coordinates and outputs but do not consistently encode query method, rationale, timestamp or model version as row-level fields.

6. **Scenario labels are illustrative**  
   The pedagogical descriptions should not be interpreted as real radiation, medical, industrial or production data.

7. **Potential notebook-copy inconsistencies**  
   Narrative text and code should be reconciled before release. For example, the observed data establishes F6 as five-dimensional even if surrounding prose refers generically to a different number of hyperparameters.

8. **No independent noise field**  
   If a function is described as noisy, the stored table still records only one scalar response per query. The dataset does not separately identify aleatoric noise, measurement error or repeated-evaluation variance.

9. **Best observed is not global optimum**  
   A high incumbent, repeated local success or boundary maximum does not prove global optimality.

---

## 5. Recommended Use and Stewardship

### 5.1 Intended uses

The dataset is suitable for:

- reproducing the documented sequential optimisation campaign;
- teaching Bayesian and black-box optimisation;
- comparing acquisition functions under a small evaluation budget;
- fitting function-specific surrogate models;
- analysing exploration–exploitation trade-offs;
- studying how adaptive sampling changes a dataset over time;
- evaluating duplicate prevention, boundary handling and local-refinement policies;
- demonstrating transparent ML project documentation.

A particularly valuable use is **policy replay**: fit a model using observations available up to round `t`, reproduce or replace the candidate-generation policy, and compare the proposed query with the one actually submitted.

### 5.2 Evaluation guidance

Because the true global optima are unknown, evaluation should focus on sequential and within-function measures such as:

- best observed value after each round;
- improvement over the initial incumbent;
- simple regret relative to the best value available in the recorded snapshot;
- number of rounds required to establish a new incumbent;
- distance between successive queries;
- fraction of queries allocated to local versus global search;
- predictive calibration on held-out historical observations, with caution about temporal leakage.

Raw outputs should not be averaged across functions. Where aggregate reporting is necessary, first convert each function to a comparable score, such as normalised improvement over its initial range or rank-based performance.

### 5.3 Uses to avoid

The dataset should not be used:

- as an unbiased sample of any complete input domain;
- for causal inference;
- to infer physical, medical or operational relationships from the illustrative scenarios;
- to train a general-purpose model across all functions using raw `y`;
- as evidence that a global optimum has been found;
- for safety-critical or high-stakes deployment;
- to benchmark large-data neural models without acknowledging the tiny sample sizes;
- to reverse-engineer or redistribute hidden course functions.

### 5.4 Ethical and practical considerations

The dataset contains no personal data, protected attributes or confidential communications. Privacy and consent risks are therefore minimal.

The main ethical consideration is **misrepresentation**. Because the notebooks use realistic scenarios, readers could mistakenly believe the data validates decisions in radiation detection, drug development, logistics or industrial chemistry. The repository should state clearly that the functions are synthetic pedagogical black boxes.

A second consideration is overconfidence. Surrogate confidence intervals depend on modelling assumptions and are not proof that unexplored regions are safe or inferior. Documentation should avoid phrases such as “the optimum is” unless a ground-truth optimum is provided; “best observed point” is the appropriate term.

### 5.5 Distribution and licensing

The dataset may be distributed with the project repository, subject to the terms governing course-provided material.

A release should distinguish:

- student-authored code and documentation;
- course-provided initial arrays;
- returned query responses;
- hidden objective definitions, which are not part of the dataset and should not be redistributed.

This datasheet does not assert a licence on behalf of Imperial College. The repository owner should add an explicit `LICENSE` and, where necessary, a note limiting redistribution of course-provided files.

### 5.6 Reproducibility requirements

A reproducible release should include:

- immutable initial input and output arrays;
- append-only query/response logs;
- cumulative per-function tables;
- the query round and source for every row;
- exact notebook or script versions;
- package versions;
- random seeds used for stochastic candidate generation;
- a script that rebuilds every cumulative dataset;
- a script that regenerates the summary statistics in this document.

The supplied notebooks frequently use fixed random states or seeds. This improves repeatability, but reproducibility still depends on the exact data snapshot, candidate pool, library versions and optimisation settings.

### 5.7 Maintenance and versioning

The dataset should be treated as a versioned experimental record.

Recommended version scheme:

```text
v0.x  — intermediate weekly snapshots
v1.0  — final capstone submission
v1.1+ — documentation or metadata corrections that do not alter raw responses
v2.0  — materially extended optimisation campaign
```

Raw observations should never be rewritten in place. Corrections should be recorded in a changelog with:

- affected function and observation ID;
- previous value;
- corrected value;
- reason;
- date;
- person making the correction.

### 5.8 Relationship to other repository documents

| Document | Responsibility |
|---|---|
| `README.md` | Explains the project, installation, repository navigation and headline results |
| `DATASHEET.md` | Documents the data, provenance, structure, quality, biases and permitted uses |
| `MODEL_CARD.md` | Documents surrogate models, acquisition policies, assumptions, evaluation and model risks |
| Function notebooks | Preserve exploratory analysis, query decisions, plots and round-by-round evidence |
| Source code | Implements data loading, modelling, acquisition and validation |
| Changelog / release notes | Records updates to data and documentation |

This division prevents duplication and makes the repository easier to audit.

### 5.9 Future improvements

The highest-value improvements are not necessarily more prose. They are better structured evidence:

1. **Synchronise all function notebooks to one final checkpoint.**
2. **Export cumulative CSV or Parquet files** rather than relying on rendered HTML tables.
3. **Add row-level provenance and query rationale.**
4. **Create a machine-readable manifest** containing dimensions, bounds, row counts and file hashes.
5. **Automate quality checks** for bounds, missing values, duplicates and schema consistency.
6. **Generate this datasheet's statistics from code** so the document cannot drift from the data.
7. **Record incumbent history** and the acquisition method used at each round.
8. **Add data-version hashes** to the Model Card and experimental reports.
9. **Preserve failed and poor queries** as first-class evidence rather than hiding them.
10. **Publish a compact data dictionary** alongside the cumulative files.

---

## Release Checklist

Before committing this document as the final repository datasheet, confirm:

- [ ] All eight notebooks reflect the intended final round.
- [ ] The row counts and output ranges have been regenerated.
- [ ] The best-observed points match the final cumulative datasets.
- [ ] Course-provided and student-generated rows are distinguishable.
- [ ] No raw response has been transformed or silently removed.
- [ ] The repository licence covers the material being distributed.
- [ ] `README.md`, `DATASHEET.md` and `MODEL_CARD.md` link to one another.
- [ ] A changelog records any correction made after submission.

---

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). *Datasheets for Datasets*. **Communications of the ACM, 64**(12), 86–92.

---

## Evidence Base for This Datasheet

This document was prepared from the supplied per-function HTML notebook exports:

- `Function_1 (16)(8).html`
- `Function_2 (8)(10).html`
- `Function_3 (4)(10).html`
- `Function_4 (4)(11).html`
- `Function_5 (5)(5).html`
- `Function_6 (4)(11).html`
- `Function_7 (2)(11).html`
- `Function_8 (2)(11).html`

The attached peer datasheets were used as structural comparators. Where those documents conflicted with the supplied cumulative HTML tables, this datasheet follows the values directly observed in the HTML snapshot and states the snapshot scope explicitly.
