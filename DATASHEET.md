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
