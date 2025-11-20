#!/usr/bin/env python3
"""
Script to create a new project that uses pySMC as a dependency

Usage:
    python create_project.py <project_name> [--path <parent_dir>]

Example:
    python create_project.py proj_40barTruss --path ../my-projects
"""

import argparse
from pathlib import Path
import sys


PYPROJECT_TEMPLATE = """[project]
name = "{project_name}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
requires-python = ">=3.10"
authors = [{{ name = "Your Name", email = "you@example.com" }}]

dependencies = [
    "pySMC",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "jupyter>=1.0",
    "ruff>=0.6",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.uv]
package = true
"""

UV_TOML_TEMPLATE = """[tool.uv.sources]
# Points to pySMC in development mode
pySMC = {{ path = "{pysmc_relative_path}", editable = true }}
"""

GITIGNORE_TEMPLATE = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
.venv/
venv/
ENV/
env/

# Environment variables
.env
.env.local
.env.*.local

# Distribution / packaging
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*~
.DS_Store

# Project outputs
results/*.csv
results/*.json
results/*.h5
figs/*.png
figs/*.jpg
figs/*.pdf

# Keep .gitkeep files
!.gitkeep

# Logs
*.log
"""

README_TEMPLATE = """# {project_title}

{description}

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   uv sync
   # or
   pip install -e .
   ```

## Usage

```bash
python app.py
```

## Project Structure

```
{project_name}/
├── src/{package_name}/      # Source code
├── data/                     # Input data
├── results/                  # Analysis results
├── figs/                     # Generated figures
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
└── app.py                    # Main application
```

## Dependencies

- pySMC (local development mode)
- See `pyproject.toml` for full list

## Development

Run tests:
```bash
pytest tests/
```

Run analysis:
```bash
python app.py
```
"""

APP_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
{project_title}

Main application for {description}
\"\"\"

from pathlib import Path
import numpy as np
from pySMC.core import Variables, MonteCarlo
import matplotlib.pyplot as plt

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR = PROJECT_ROOT / "figs"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)


def analysis_function(params):
    \"\"\"
    Main analysis function
    
    Parameters:
    -----------
    params : dict
        Dictionary with parameter values
    
    Returns:
    --------
    dict
        Results dictionary
    \"\"\"
    # Your analysis code here
    result = {{
        'output1': params['x'] ** 2,
        'output2': params['y'] * 2,
    }}
    return result


def define_parameters():
    \"\"\"Define uncertain input parameters\"\"\"
    variables = Variables()
    
    # Add your parameters
    variables.add_uniform('x', 0, 1, label='Parameter X')
    variables.add_uniform('y', 0, 10, label='Parameter Y')
    
    return variables


def main():
    \"\"\"Main application\"\"\"
    print("="*60)
    print("{project_title}")
    print("="*60)
    
    # Define parameters
    variables = define_parameters()
    
    # Run Monte Carlo
    print("Running analysis...")
    mc = MonteCarlo(analysis_function, variables)
    results = mc.run(n_samples=100)
    
    # Save results
    results_file = RESULTS_DIR / "results.csv"
    results.to_csv(results_file)
    print(f"Results saved to {{results_file}}")
    
    # Summary
    print("\\nResults summary:")
    print(results.describe())


if __name__ == '__main__':
    main()
"""


def create_project(project_name: str, parent_dir: Path, pysmc_path: Path):
    """Create a new project structure"""
    
    # Validate project name
    if not project_name.replace('_', '').replace('-', '').isalnum():
        print(f"Error: Invalid project name '{project_name}'")
        print("Use only letters, numbers, hyphens, and underscores")
        sys.exit(1)
    
    # Create project directory
    project_dir = parent_dir / project_name
    if project_dir.exists():
        print(f"Error: Directory '{project_dir}' already exists")
        sys.exit(1)
    
    project_dir.mkdir(parents=True)
    print(f"Creating project: {project_dir}")
    
    # Create directory structure
    package_name = project_name.replace('-', '_').lower()
    (project_dir / "src" / package_name).mkdir(parents=True)
    (project_dir / "data").mkdir()
    (project_dir / "results").mkdir()
    (project_dir / "figs").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "notebooks").mkdir()
    
    # Create __init__.py files
    (project_dir / "src" / package_name / "__init__.py").touch()
    (project_dir / "tests" / "__init__.py").touch()
    
    # Create .gitkeep files
    (project_dir / "results" / ".gitkeep").touch()
    (project_dir / "figs" / ".gitkeep").touch()
    (project_dir / "data" / ".gitkeep").touch()
    
    # Calculate relative path from project to pySMC
    pysmc_relative = Path('..') / pysmc_path.name
    
    # Create configuration files
    description = f"{project_name.replace('_', ' ').replace('-', ' ').title()} project using pySMC"
    project_title = project_name.replace('_', ' ').replace('-', ' ').title()
    
    # pyproject.toml
    (project_dir / "pyproject.toml").write_text(
        PYPROJECT_TEMPLATE.format(
            project_name=project_name,
            description=description,
        )
    )
    
    # uv.toml
    (project_dir / "uv.toml").write_text(
        UV_TOML_TEMPLATE.format(
            pysmc_relative_path=str(pysmc_relative),
        )
    )
    
    # .gitignore
    (project_dir / ".gitignore").write_text(GITIGNORE_TEMPLATE)
    
    # README.md
    (project_dir / "README.md").write_text(
        README_TEMPLATE.format(
            project_name=project_name,
            project_title=project_title,
            package_name=package_name,
            description=description,
        )
    )
    
    # app.py
    (project_dir / "app.py").write_text(
        APP_TEMPLATE.format(
            project_title=project_title,
            description=description,
        )
    )
    
    # Initialize git
    import subprocess
    try:
        subprocess.run(['git', 'init'], cwd=project_dir, check=True, capture_output=True)
        print(f"✓ Initialized git repository")
    except subprocess.CalledProcessError:
        print("⚠ Warning: Could not initialize git repository")
    
    print("\n" + "="*60)
    print("Project created successfully!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. cd {project_dir}")
    print(f"  2. uv sync              # Install dependencies")
    print(f"  3. python app.py        # Run application")
    print(f"  4. git add .")
    print(f"  5. git commit -m 'Initial commit'")
    print(f"\nProject structure:")
    print(f"  {project_dir}/")
    print(f"    ├── src/{package_name}/  # Your code here")
    print(f"    ├── data/               # Input data")
    print(f"    ├── results/            # Output results")
    print(f"    ├── figs/               # Generated figures")
    print(f"    └── app.py              # Main application")


def main():
    parser = argparse.ArgumentParser(
        description='Create a new project that uses pySMC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_project.py proj_40barTruss
  python create_project.py my-fe-project --path ~/workspace/projects
        """
    )
    parser.add_argument('project_name', help='Name of the project to create')
    parser.add_argument('--path', type=Path, default=Path('../my-projects'),
                       help='Parent directory for the project (default: ../my-projects)')
    
    args = parser.parse_args()
    
    # Get pySMC path (current directory)
    pysmc_path = Path(__file__).parent.resolve()
    
    # Get or create parent directory
    parent_dir = args.path.resolve()
    if not parent_dir.exists():
        print(f"Creating parent directory: {parent_dir}")
        parent_dir.mkdir(parents=True)
    
    create_project(args.project_name, parent_dir, pysmc_path)


if __name__ == '__main__':
    main()


