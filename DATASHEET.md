# Datasheet for the Black-Box Optimisation Dataset

> **Project:** Bayesian Black-Box Optimisation Capstone  
> **Objective:** Maximise eight unknown objective functions under a restricted evaluation budget  
> **Programme:** Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence  
> **Datasheet framework:** Adapted from *Datasheets for Datasets* (Gebru et al., 2021)

---

## 1. Motivation

- **Purpose:** This dataset supports the sequential optimisation of eight independent black-box objective functions, labelled **F1–F8**. Each function can be investigated only by submitting a valid input vector and observing the scalar response returned by the hidden evaluator.

- **Task supported:** The dataset supports Bayesian and related black-box optimisation workflows across functions ranging from two to eight dimensions. It enables analysis of surrogate modelling, acquisition strategies, exploration–exploitation trade-offs and optimisation performance under a restricted evaluation budget.

- **Creator and context:** The initial design was provided as part of a black-box optimisation challenge, while the underlying objective functions remained hidden. All subsequent query points were designed and selected by the project author using the analytical workflows, code and experiment notebooks contained in this repository.

- **Iterative Process Description:** Unlike a conventional supervised-learning dataset collected before modelling begins, this dataset was built sequentially. Each query was selected using the observations available at that stage, so later observations depend on earlier outcomes and collection order is relevant to reproducing and evaluating the optimisation process.

This datasheet documents the dataset’s provenance, composition, preparation, biases, limitations and recommended uses. Modelling choices and optimisation diagnostics are documented separately in `MODEL_CARD.md` and the function notebooks.

---

## 2. Composition

### 2.1 Unit of observation

The dataset contains observations from eight independent black-box objective functions. Each observation consists of an input vector and the corresponding evaluator response:

```math
(\mathbf{x}, y), \qquad \mathbf{x} \in [0,1]^d,\quad y \in \mathbb{R}
```

where:

- `x1, ..., xd` are continuous input coordinates;
- `d` is the dimensionality of the relevant function;
- `y` is the scalar response returned by the hidden evaluator;
- the optimisation objective is to maximise `y`.

### 2.2 Dataset summary

| Property | Value |
|---|---|
| Objective functions | 8 |
| Dimensionality | 2 to 8 input dimensions |
| Input domain | Unit hypercube, `[0,1]^d` |
| Optimisation direction | Maximise `y` |
| Initial observations | 175 |
| Sequentially acquired observations | 104 |
| Total observations | 279 |
| Data format | NumPy arrays of floating-point inputs and outputs |
| Experimental record | Per-function notebooks and cumulative tables |
| Data sensitivity | Synthetic numerical data containing no personal information |

### 2.3 Function-level composition

The table below summarises the final cumulative dataset. Observation counts and response ranges include both the initial and sequentially acquired observations.

| Function | Dim. | Initial | Sequential | Total | Cumulative observed `y` range | Observed characteristics |
|---|---:|---:|---:|---:|---:|---|
| F1 | 2 | 10 | 13 | 23 | `[-0.003606063, 8.190854e-16]` | Responses concentrated extremely close to zero |
| F2 | 2 | 10 | 13 | 23 | `[-0.065624, 0.632404]` | Positive higher-value regions observed at multiple locations |
| F3 | 3 | 15 | 13 | 28 | `[-0.398926, -0.00626387]` | Negative responses; values nearer zero are preferable |
| F4 | 4 | 30 | 13 | 43 | `[-32.625660, 0.585185]` | Large negative tail with a comparatively small positive region |
| F5 | 4 | 20 | 13 | 33 | `[0.112940, 8662.4825]` | Strong observed response growth towards a boundary corner |
| F6 | 5 | 20 | 13 | 33 | `[-2.571170, -0.103406]` | Negative responses; larger values lie nearer zero |
| F7 | 6 | 30 | 13 | 43 | `[0.002701, 2.854626]` | Strong observed responses near a boundary |
| F8 | 8 | 40 | 13 | 53 | `[5.592193, 9.99309]` | Comparatively narrow observed response range |

The reported ranges describe only the responses contained in the dataset. They do not represent the theoretical ranges of the hidden functions or demonstrate that their global optima were identified.

### 2.4 Completeness and coverage

The dataset is complete with respect to the initial observations and all submitted queries. It does not provide complete coverage of the underlying search spaces.

Because each function was evaluated under a restricted query budget, large regions remain unexplored, particularly for the higher-dimensional functions F6–F8. Sequential observations were selected adaptively and are therefore not uniformly or independently sampled.

---

## 3. Collection Process

### 3.1 Provenance

The dataset combines two sources of observations:

- **Initial observations:** Input vectors and corresponding outputs provided at the start of the challenge. These formed the warm-start dataset for each function.
- **Sequentially acquired observations:** Additional input vectors selected by the project author during the optimisation process. Each submitted vector was evaluated by the hidden evaluator, which returned one scalar response.

The underlying objective functions remained undisclosed throughout the collection process.

### 3.2 Sequential acquisition

Sequential observations were collected through an iterative weekly process. At each stage, the cumulative observations were analysed to identify the next query point. Selection was informed by exploratory analysis, surrogate-model outputs, acquisition strategies and function-specific diagnostics.

The selected input vector was submitted to the hidden evaluator, and the returned response was validated and appended to the cumulative dataset for the relevant function.

Because each query was selected using the information available at that stage, later observations depend on earlier results. Where available, the observation source and query round should therefore be retained.

### 3.3 Sampling strategy

The sampling strategy generally evolved from broad exploration towards more targeted refinement:

- **Earlier rounds:** Space-filling exploration, interior points and boundary tests.
- **Middle rounds:** A balance between exploration and exploitation, informed by observed response patterns and surrogate-model estimates.
- **Later rounds:** Local refinement around regions associated with higher observed responses.

This progression varied by function and does not imply that unexplored regions were inferior or that a global optimum was identified.

---

## 4. Data Preparation and Quality

### 4.1 Preparation and validation

Observations were retained as returned by the hidden evaluator. No preprocessing, transformation, outlier removal or manual labelling was applied. Before being appended, each observation was checked for valid dimensionality, input coordinates within `[0,1]`, duplicate query points and a scalar response.

### 4.2 Data preservation

All valid evaluator responses were retained, including negative values, near-zero values, poor exploratory results, boundary observations and repeated outputs returned at different input points. No observation was removed or modified based on its magnitude or optimisation performance.

---

## 5. Biases and Limitations

### 5.1 Sampling biases

- **Adaptive and policy-dependent sampling:** Query points were deliberately selected because they appeared informative or promising. The dataset therefore over-represents regions favoured by the optimisation strategy and under-represents unexplored regions.
- **Temporal dependence:** Later observations were selected using earlier results. Collection order must be preserved when evaluating the sequential process.
- **Uneven dimensional coverage:** The restricted query budget provides much sparser effective coverage for higher-dimensional functions than for lower-dimensional functions.
- **Boundary concentration:** Several strong observations occur near boundaries or corners. This does not establish that the true optimum lies on a boundary.

These characteristics are appropriate for sequential optimisation research but prevent the dataset from being treated as a representative sample of each complete input domain.

### 5.2 Limitations

- **Restricted sample size:** Each function contains only 23–53 observations, limiting statistical certainty.
- **Unknown ground truth:** The analytical functions and true global optima are unavailable.
- **Sparse high-dimensional coverage:** F6–F8 cannot be mapped comprehensively using the available observations.
- **Illustrative scenarios:** Any physical, medical, industrial or operational interpretations are pedagogical only.
- **No independent noise measure:** Each query records one scalar response, without repeated evaluations or a separate noise field.
- **Best observed is not globally optimal:** A high incumbent or repeated local success does not prove that the global optimum has been found.

---

## 6. Recommended Use

### 6.1 Intended uses

The dataset is suitable for:

- reproducing the documented sequential optimisation campaign;
- teaching Bayesian and black-box optimisation;
- fitting function-specific surrogate models;
- comparing acquisition strategies under a restricted evaluation budget;
- analysing exploration–exploitation behaviour and adaptive sampling.

Evaluation should be performed within each function because their response scales are not directly comparable. Appropriate measures include the best-observed value by round and improvement over the initial incumbent.

### 6.2 Uses to avoid

The dataset should not be used:

- as an unbiased or complete representation of any function’s input domain;
- for causal inference;
- to infer real physical, medical, industrial or operational relationships from illustrative scenarios;
- to train a single model across all functions using unadjusted raw outputs;
- as evidence that a global optimum has been identified;
- for safety-critical or high-stakes deployment.

### 6.3 Ethical considerations

The dataset contains no personal data or protected attributes. The main risk is misrepresentation: illustrative scenarios should not be interpreted as evidence from real systems.

Documentation should use the term **best observed point** rather than **optimum** unless a ground-truth optimum is provided.

---

## 7. Distribution, Reproducibility and Maintenance

### 7.1 Distribution and licensing

The dataset may be distributed with the project repository, subject to the terms governing course-provided material. A release should distinguish between:

- project-authored code and documentation;
- course-provided initial arrays;
- evaluator responses obtained through submitted queries;
- hidden objective definitions, which are not part of the dataset.

This datasheet does not assert a licence on behalf of Imperial College. The repository licence should clearly state the terms applicable to project-authored and course-provided material.

### 7.2 Reproducibility and maintenance

A reproducible release should include the initial arrays, cumulative per-function tables, query order, source information and the code or notebooks used to regenerate the dataset summaries.

Raw observations should remain unchanged. Any corrections or additions should be versioned and recorded in the project changelog.

### 7.3 Related repository documents

| Document | Responsibility |
|---|---|
| `README.md` | Project overview, installation, repository navigation and headline results |
| `DATASHEET.md` | Dataset provenance, composition, quality, biases and permitted uses |
| `MODEL_CARD.md` | Surrogate models, acquisition policies, assumptions, evaluation and model risks |
| Function notebooks | Exploratory analysis, query decisions, plots and round-by-round evidence |
| Source code | Data loading, modelling, acquisition and validation |

---

## 8. References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). *Datasheets for Datasets*. **Communications of the ACM, 64**(12), 86–92.
