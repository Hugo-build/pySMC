"""
Test script for ParametricVariable and ConfigMapper functionality
"""

import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.Variables import ParametricVariable, ConfigMapper, create_parametric_study


def test_simple_injection():
    """Test basic path injection"""
    print("\n" + "="*70)
    print("TEST 1: Simple path injection")
    print("="*70)
    
    # Create a simple config
    config = {
        'param_a': 10,
        'param_b': 20,
        'nested': {
            'value': 100
        }
    }
    
    # Define variable with path
    var = ParametricVariable(
        name="a",
        kind="uniform",
        params={"low": 5, "high": 15},
        config_paths=["param_a"]
    )
    
    # Inject value
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"a": 12})
    
    print(f"Original param_a: {config['param_a']}")
    print(f"Modified param_a: {new_config['param_a']}")
    assert new_config['param_a'] == 12, "Simple injection failed"
    assert config['param_a'] == 10, "Original config was modified!"
    print("✓ Test passed")


def test_indexed_injection():
    """Test array index access"""
    print("\n" + "="*70)
    print("TEST 2: Indexed path injection")
    print("="*70)
    
    config = {
        'loads': [0, 0, 0, 0, 1000, 0, 0],
        'values': np.array([1, 2, 3, 4, 5])
    }
    
    var = ParametricVariable(
        name="load",
        kind="fixed",
        params={"value": 1500},
        config_paths=["loads[4]"]
    )
    
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"load": 1500})
    
    print(f"Original loads[4]: {config['loads'][4]}")
    print(f"Modified loads[4]: {new_config['loads'][4]}")
    assert new_config['loads'][4] == 1500, "Indexed injection failed"
    print("✓ Test passed")


def test_nested_injection():
    """Test nested dictionary access"""
    print("\n" + "="*70)
    print("TEST 3: Nested path injection")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'E': 200e3, 'A': 100},
            {'id': 1, 'E': 200e3, 'A': 100},
            {'id': 2, 'E': 200e3, 'A': 100},
        ]
    }
    
    var = ParametricVariable(
        name="E_elem1",
        kind="uniform",
        params={"low": 190e3, "high": 210e3},
        config_paths=["elements[1].E"]
    )
    
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"E_elem1": 205e3})
    
    print(f"Original elements[1]['E']: {config['elements'][1]['E']}")
    print(f"Modified elements[1]['E']: {new_config['elements'][1]['E']}")
    assert new_config['elements'][1]['E'] == 205e3, "Nested injection failed"
    assert new_config['elements'][0]['E'] == 200e3, "Wrong element modified!"
    print("✓ Test passed")


def test_wildcard_injection():
    """Test wildcard path (apply to all elements)"""
    print("\n" + "="*70)
    print("TEST 4: Wildcard path injection")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'E': 200e3, 'A': 100},
            {'id': 1, 'E': 200e3, 'A': 100},
            {'id': 2, 'E': 200e3, 'A': 100},
        ]
    }
    
    var = ParametricVariable(
        name="E_all",
        kind="uniform",
        params={"low": 190e3, "high": 210e3},
        config_paths=["elements[*].E"]
    )
    
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"E_all": 210e3})
    
    print("Original E values:", [elem['E'] for elem in config['elements']])
    print("Modified E values:", [elem['E'] for elem in new_config['elements']])
    
    for elem in new_config['elements']:
        assert elem['E'] == 210e3, f"Element {elem['id']} not updated"
    print("✓ Test passed - all elements updated")


def test_multiple_paths():
    """Test variable with multiple paths"""
    print("\n" + "="*70)
    print("TEST 5: Variable with multiple paths")
    print("="*70)
    
    config = {
        'param1': 10,
        'param2': 20,
        'params': [1, 2, 3]
    }
    
    var = ParametricVariable(
        name="shared_value",
        kind="fixed",
        params={"value": 42},
        config_paths=["param1", "param2", "params[0]"]
    )
    
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"shared_value": 42})
    
    print(f"Modified param1: {new_config['param1']}")
    print(f"Modified param2: {new_config['param2']}")
    print(f"Modified params[0]: {new_config['params'][0]}")
    
    assert new_config['param1'] == 42, "param1 not updated"
    assert new_config['param2'] == 42, "param2 not updated"
    assert new_config['params'][0] == 42, "params[0] not updated"
    print("✓ Test passed - all paths updated")


def test_callable_path():
    """Test callable path injection"""
    print("\n" + "="*70)
    print("TEST 6: Callable path injection")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'rho': 7.85e-6},
            {'id': 1, 'rho': 7.85e-6},
        ]
    }
    
    def set_all_densities(cfg, value):
        for elem in cfg['elements']:
            elem['rho'] = value
    
    var = ParametricVariable(
        name="density",
        kind="normal",
        params={"mean": 7.85e-6, "std": 0.1e-6},
        config_paths=[set_all_densities]
    )
    
    mapper = ConfigMapper([var])
    new_config = mapper.inject(config, {"density": 8.0e-6})
    
    print("Modified densities:", [elem['rho'] for elem in new_config['elements']])
    
    for elem in new_config['elements']:
        assert elem['rho'] == 8.0e-6, f"Element {elem['id']} density not updated"
    print("✓ Test passed - callable path worked")


def test_sampling_and_injection():
    """Test automatic sampling with create_parametric_study"""
    print("\n" + "="*70)
    print("TEST 7: Automatic sampling and injection")
    print("="*70)
    
    base_config = {
        'E': 200e3,
        'load': 1000,
        'elements': [
            {'id': 0, 'A': 100},
            {'id': 1, 'A': 100},
        ]
    }
    
    variables = [
        ParametricVariable(
            name="E", kind="uniform",
            params={"low": 190e3, "high": 210e3},
            config_paths=["E"]
        ),
        ParametricVariable(
            name="load", kind="uniform",
            params={"low": 500, "high": 1500},
            config_paths=["load"]
        ),
    ]
    
    configs, samples = create_parametric_study(
        base_config=base_config,
        variables=variables,
        n_samples=5,
        seed=42
    )
    
    print(f"Generated {len(configs)} configs")
    print(f"Sample array shape: {samples.shape}")
    print(f"First config E: {configs[0]['E']:.2f}")
    print(f"First config load: {configs[0]['load']:.2f}")
    
    assert len(configs) == 5, "Wrong number of configs"
    assert samples.shape == (5, 2), "Wrong sample array shape"
    assert configs[0]['E'] == samples[0, 0], "Config doesn't match sample"
    assert configs[0]['load'] == samples[0, 1], "Config doesn't match sample"
    print("✓ Test passed")


def test_multiple_variables():
    """Test injection with multiple variables"""
    print("\n" + "="*70)
    print("TEST 8: Multiple variable injection")
    print("="*70)
    
    config = {
        'E': 200e3,
        'A': 100,
        'load': 1000,
        'elements': [{'id': 0, 'L': 1000}]
    }
    
    variables = [
        ParametricVariable(
            name="E", kind="uniform",
            params={"low": 190e3, "high": 210e3},
            config_paths=["E"]
        ),
        ParametricVariable(
            name="A", kind="uniform",
            params={"low": 90, "high": 110},
            config_paths=["A"]
        ),
        ParametricVariable(
            name="load", kind="uniform",
            params={"low": 500, "high": 1500},
            config_paths=["load"]
        ),
    ]
    
    mapper = ConfigMapper(variables)
    new_config = mapper.inject(config, {
        "E": 205e3,
        "A": 95,
        "load": 1200
    })
    
    print(f"Modified E: {new_config['E']}")
    print(f"Modified A: {new_config['A']}")
    print(f"Modified load: {new_config['load']}")
    
    assert new_config['E'] == 205e3, "E not updated"
    assert new_config['A'] == 95, "A not updated"
    assert new_config['load'] == 1200, "load not updated"
    print("✓ Test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Running ParametricVariable and ConfigMapper tests")
    print("#"*70)
    
    test_simple_injection()
    test_indexed_injection()
    test_nested_injection()
    test_wildcard_injection()
    test_multiple_paths()
    test_callable_path()
    test_sampling_and_injection()
    test_multiple_variables()
    
    print("\n" + "#"*70)
    print("# ALL TESTS PASSED ✓")
    print("#"*70 + "\n")


if __name__ == "__main__":
    run_all_tests()

