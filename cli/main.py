from __future__ import annotations
import typer, pandas as pd
from typing import Optional
from ..templates.base import all_cases, get
from ..core.monte_carlo import run_monte_carlo
from ..core.samplers import SamplerKind
from ..io.loader import load_case_from_file, build_case_from_config, get_sampling_from_config

app = typer.Typer(help="Monte Carlo runner for case studies.")

@app.command()
def list_cases():
    for name in all_cases():
        typer.echo(name)

@app.command()
def run(
    case: str = typer.Argument(..., help="Case name (see `list-cases`)."),
    n: int = typer.Option(256, help="Number of samples."),
    sampler: SamplerKind = typer.Option("sobol", help="Sampler: random|lhs|sobol"),
    seed: Optional[int] = typer.Option(None, help="Random seed."),
    out: Optional[str] = typer.Option(None, help="Path to write CSV results."),
):
    c = get(case)
    res = run_monte_carlo(c.evaluate, c.variables, n=n, sampler=sampler, seed=seed)
    df = pd.concat([res.inputs, res.outputs], axis=1)
    if out:
        df.to_csv(out, index=False)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(df.head().to_string())

@app.command()
def run_config(
    config: str = typer.Argument(..., help="Path to JSON/YAML config file."),
    out: Optional[str] = typer.Option(None, help="Path to write CSV results."),
):
    cfg = load_case_from_file(config)
    case = build_case_from_config(cfg)
    samp = get_sampling_from_config(cfg)
    if "samples" in samp:
        res = run_monte_carlo(case.evaluate, case.variables, samples=samp["samples"])
    else:
        res = run_monte_carlo(case.evaluate, case.variables,
                              n=samp.get("n",256), sampler=samp.get("sampler","sobol"), seed=samp.get("seed",None))
    df = pd.concat([res.inputs, res.outputs], axis=1)
    if out:
        import pathlib
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(df.head().to_string())

def main():
    app()

if __name__ == "__main__":
    main()
