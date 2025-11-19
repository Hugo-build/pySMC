## LLM Agent Integration Outline (pySMC)

This note sketches which functions to expose to an LLM agent and whether to wrap them behind an MCP server. It focuses on safe, composable tools to let an agent run FE studies, sampling, sensitivity analysis, and surrogate modeling using existing pySMC modules.

### Goals
- Enable an LLM agent to: build/modify FE configs, run parametric sweeps, perform sensitivity analysis, fit/query surrogates, and manage artifacts (plots/CSV).
- Provide explicit and safe tool contracts with clear inputs/outputs and path whitelisting.

### Approaches
- In-process tool functions: Direct Python-callable tool registry (simple to start; best for local notebooks/scripts).
- MCP server: Standardized tool interface consumable by MCP-aware clients (IDEs, copilots). Better isolation, auditing, and permission prompts.

Recommendation: start with an in-process tool registry for rapid iteration; in parallel, design tool schemas so they can be lifted into an MCP server with minimal changes.

---

## Core Data Schemas (JSON-like)

- FEConfig
  - nodes: Array<Node>
  - elements: Array<Element>
  - fixed_dofs: Array<int>
  - loads: Array<number>

- Node
  - id: int
  - coords: [number, number, number]  // mm

- Element
  - id: int
  - node_i: int
  - node_j: int
  - E: number  // modulus
  - A: number  // area
  - rho: number

- VariableSpec
  - name: string
  - kind: "uniform" | "normal" | "lognormal" | "fixed"
  - params: object  // e.g., {low, high} or {mean, std}
  - targets?: Array<{ doc: string, path: string }>

---

## Candidate Tools to Expose

### FE and Parametric Study
- fe.load_40bar_truss()
  - Loads `40barTruss/nodes.csv` and `elements.csv` into FEConfig.
  - Out: FEConfig

- fe.solve_static(fe_config)
  - Wraps `solve_FE_static` from `exp_40barTruss.py`.
  - In: FEConfig
  - Out: { displacement: number[], max_displacement: number }

- fe.plot_deformation(fe_config, displacement, scale?)
  - Wraps `plot_displacements` for visualization.
  - Out: { path: string }  // saved figure under `figs/`

### Variable Definition and Injection
- vars.create_variable(name, kind, params)
  - Returns a VariableSpec

- vars.add_target(variable, doc, path)
  - Adds an injection target (e.g., `elements[*].E`, `loads[29]`).

- vars.make_set(variables)
  - Creates `VariableSet`

- vars.inject(configs, values)
  - Wraps `VariableSet.inject_values` to produce modified configs
  - In: { configs: {FE: FEConfig}, values: Record<string, number> }
  - Out: { configs: {FE: FEConfig} }

- vars.sample_configs(configs, var_set, n, seed?)
  - Wraps `VariableSet.sample_configs`
  - Out: { sampled_configs: Array<{FE: FEConfig}>, samples: number[][] }

- vars.to_salim_problem(var_set)
  - Wraps `VariableSet.to_SAlib()`

### Sampling and DoE
- sampling.sample_inputs(var_set, n, kind?, seed?)
  - Wraps `core.Samplers.sample_inputs` (random|lhs|sobol)

### Sensitivity Analysis
- sa.sobol_indices(problem, y)
  - Wraps SALib’s `analyze.sobol` call with the `problem` from `to_SAlib()`.
  - In: problem, y: number[]
  - Out: standard Sobol dict (S1, ST, etc.)

### Surrogate Modeling (GP)
- gp.create_kernel(kind, init_from?)
  - RBF, Matern32, Matern52; produce initial log hyperparameters (sf, ls)

- gp.create(model_cfg)
  - Builds `core.GPax.GaussianProcess` with kernel, noise (log_sn2), jitter

- gp.fit(gp, X, y, opt_config?)
  - Wraps `GaussianProcess.fit` and `optSetup`
  - opt_config: { optimizer: 'adam'|'lbfgs'|'sgd', steps, lr, log_every, verbose, tol_*? }
  - Out: fitted gp (serializable reference)

- gp.predict(gp, X)
  - Wraps `GaussianProcess.predict`
  - Out: { mean: number[], std: number[] }

- data.scale(X, y)
  - Wraps `core.DataWash.scale_data`
  - Out: { X_scaled, y_scaled, x_scaler, y_scaler }

- data.train_test_split(X, y, test_size?, random_state?)
  - Wraps `core.DataWash.train_test_split`

### Adaptive Learning Utilities
- adapt.weight.calc(X_old, y_old, X_new, y_new, predict_fn_kind?, params?)
  - Wraps `calc_upd_weight` with default `SizeNoveltyWeight`

- adapt.weight.combine(X_old, y_old, X_new, y_new, weight, seed?)
  - Wraps `combine_weighted_data` to produce a merged dataset

### IO and CLI
- io.load_case_config(path)
  - Wraps `io.loader.load_case_from_file`

- io.build_case(cfg)
  - Wraps `io.loader.build_case_from_config`

- cli.run(case, n?, sampler?, seed?, out?)
  - Wraps `cli.main.run` programmatically; returns dataframe head + path if written

---

## General demo() Entrypoint (agent-friendly)

- agent.demo(case, n?, sampler?, seed?, run_sa?, run_gp?)
  - Location: `core/demo.py`
  - Inputs:
    - `case`: any object with `variables: VariableSet` and `evaluate(x) -> Dict[str,float]`
    - `n`, `sampler`: sample size and strategy (random|lhs|sobol)
    - `seed`: RNG seed
    - `run_sa`: compute Sobol indices (if SALib available)
    - `run_gp`: fit a simple GP surrogate (if JAX available)
  - Outputs: `{inputs, outputs, primary_key, y, sa?, gp?, artifacts[]}`
  - Purpose: provide a domain-agnostic workflow so the agent can execute a standard pipeline without FE-specific logic.

The FE demo in `exp_40barTruss.py` can be adapted to provide a `case` that implements `evaluate`, allowing reuse of this general demo.

---

## Abstract Tool Interfaces (domain-agnostic)

To make these tools reusable beyond a specific FE problem, we define abstract interfaces the agent can rely on (implemented in `core/AgentAPI.py`):

- CaseLike
  - properties: `variables: VariableSet`
  - method: `evaluate(x: np.ndarray) -> Dict[str, float>`
  - Works for algebraic, ODE, FE, external simulators.

- ConfigInjector
  - `inject(configs, var_set, values) -> configs`
  - `sample_configs(configs, var_set, n_samples, seed?) -> (configs_list, samples)`

- SamplerService
  - `sample_inputs(var_set, n, kind='random'|'lhs'|'sobol', seed?) -> np.ndarray`

- SurrogateModel
  - `fit(X, y, opt_config?) -> SurrogateModel`
  - `predict(X) -> (mean, std)`
  - `get_params_tree() / set_params_tree(tree)`

- SensitivityAnalyzer
  - `sobol(problem, y) -> Dict`

- AcquisitionFunction
  - `evaluate(mean, std, **kwargs) -> score`

These interfaces let the MCP server register tools against abstract capabilities rather than concrete implementations. The FE example simply binds to these via thin adapters.

---

## Guardrails and Policies
- File access: whitelist project `figs/`, `40barTruss/` and disallow arbitrary paths by default.
- Runtime bounds: max samples per call; max steps for optimizers; budgets per tool.
- Determinism: accept optional `seed` for all stochastic functions.
- Resource control: expose verbosity flags; avoid interactive prompts.
- Error surfaces: return structured errors with helpful remediation hints.

---

## MCP Server Plan

### Transport and Serving
- Language: Python server
- Protocol: Model Context Protocol (MCP) with tool invocation over stdio or WebSocket
- Packaging: lightweight runner script (e.g., `python -m pysmc.mcp_server`)

### Tool Registration
- Define each tool with name, description, JSON schema for params/result.
- Map to safe wrappers that call into pySMC modules; no direct eval.

### Types and Schemas
- Provide JSON Schemas for `FEConfig`, `VariableSpec`, `VariableSetRef`, `OptimizerConfig`, etc.
- Use IDs/handles for large objects (e.g., GP instances) with an object registry and TTL.

### State Management
- Short-lived handles for fitted models and scalers; explicit save/load endpoints later.
- Idempotent operations where possible (e.g., repeated `plot_deformation` overwrites or versions files with timestamps).

### Security and Permissions
- Path whitelisting and deny-all defaults.
- Rate limiting and max-size inputs (e.g., samples, matrix sizes).
- Optional approval prompts for expensive tasks (client-driven via MCP).

### Observability
- Structured logs per tool call (inputs metadata, timing, outputs summary).
- Return artifact paths for plots/CSVs; clients can fetch as needed.

---

## Minimal First Set (MVP)
1) fe.load_40bar_truss, fe.solve_static, fe.plot_deformation
2) vars.create_variable, vars.add_target, vars.make_set, vars.inject, vars.sample_configs
3) sampling.sample_inputs
4) gp.create, gp.fit, gp.predict, data.scale, data.train_test_split
5) sa.sobol_indices

This set supports the full workflow demonstrated in `exp_40barTruss.py` end-to-end, while keeping the surface area small.

---

## Next Steps
- Implement in-process tool registry with the signatures above.
- Add a thin MCP server exposing the same tools and schemas.
- Add examples and tests for typical agent tasks (FE sweep, SA, GP training).
- Incrementally broaden coverage (IO cases, adaptive loops, pool/ensembles).


