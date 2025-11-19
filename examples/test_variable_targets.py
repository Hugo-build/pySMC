"""
Quick test of Variable.targets functionality
"""
import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.Variables import Variable, VariableSet, inject_single_config

def test_basic_injection():
    """Test basic path injection"""
    print("="*70)
    print("TEST 1: Basic injection with single target")
    print("="*70)
    
    config = {
        'param_a': 10,
        'param_b': 20
    }
    
    var = Variable(
        name="a",
        kind="uniform",
        params={"low": 5, "high": 15}
    )
    var.add_target(doc="test", path="param_a")
    
    var_set = VariableSet(variables=[var])
    configs = {"test": config}
    
    new_configs = var_set.inject_values(configs, {"a": 12})
    
    assert config['param_a'] == 10, "Original config was modified!"
    assert new_configs["test"]['param_a'] == 12, "Injection failed!"
    print("✓ PASSED: Basic injection works\n")


def test_indexed_path():
    """Test array indexing"""
    print("="*70)
    print("TEST 2: Indexed path injection")
    print("="*70)
    
    config = {
        'loads': [0, 0, 0, 1000, 0, 0]
    }
    
    var = Variable(name="F", kind="fixed", params={"value": 1500})
    var.add_target(doc="sim", path="loads[3]")
    
    var_set = VariableSet(variables=[var])
    new_configs = var_set.inject_values({"sim": config}, {"F": 1500})
    
    assert new_configs["sim"]["loads"][3] == 1500, "Indexed injection failed!"
    assert config['loads'][3] == 1000, "Original was modified!"
    print("✓ PASSED: Indexed path works\n")


def test_nested_path():
    """Test nested dictionary paths"""
    print("="*70)
    print("TEST 3: Nested path injection")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'E': 200e3, 'A': 100},
            {'id': 1, 'E': 200e3, 'A': 100},
            {'id': 2, 'E': 200e3, 'A': 100},
        ]
    }
    
    var = Variable(name="E1", kind="uniform", params={"low": 190e3, "high": 210e3})
    var.add_target(doc="FE", path="elements[1].E")
    
    var_set = VariableSet(variables=[var])
    new_configs = var_set.inject_values({"FE": config}, {"E1": 205e3})
    
    assert new_configs["FE"]["elements"][1]['E'] == 205e3, "Nested injection failed!"
    assert new_configs["FE"]["elements"][0]['E'] == 200e3, "Wrong element modified!"
    assert config['elements'][1]['E'] == 200e3, "Original was modified!"
    print("✓ PASSED: Nested path works\n")


def test_wildcard_path():
    """Test wildcard paths"""
    print("="*70)
    print("TEST 4: Wildcard path injection")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'E': 200e3},
            {'id': 1, 'E': 200e3},
            {'id': 2, 'E': 200e3},
        ]
    }
    
    var = Variable(name="E_all", kind="uniform", params={"low": 190e3, "high": 210e3})
    var.add_target(doc="FE", path="elements[*].E")
    
    var_set = VariableSet(variables=[var])
    new_configs = var_set.inject_values({"FE": config}, {"E_all": 210e3})
    
    for i, elem in enumerate(new_configs["FE"]["elements"]):
        assert elem['E'] == 210e3, f"Element {i} not updated!"
    
    for i, elem in enumerate(config['elements']):
        assert elem['E'] == 200e3, f"Original element {i} was modified!"
    
    print("✓ PASSED: Wildcard path works\n")


def test_multiple_targets():
    """Test variable with multiple targets"""
    print("="*70)
    print("TEST 5: Multiple targets per variable")
    print("="*70)
    
    config = {
        'elements': [
            {'id': 0, 'A': 100},
            {'id': 1, 'A': 100},
            {'id': 2, 'A': 100},
        ]
    }
    
    var = Variable(name="A_group", kind="uniform", params={"low": 90, "high": 110})
    var.add_target(doc="FE", path="elements[0].A")
    var.add_target(doc="FE", path="elements[1].A")
    
    var_set = VariableSet(variables=[var])
    new_configs = var_set.inject_values({"FE": config}, {"A_group": 95})
    
    assert new_configs["FE"]["elements"][0]['A'] == 95, "Target 0 not updated!"
    assert new_configs["FE"]["elements"][1]['A'] == 95, "Target 1 not updated!"
    assert new_configs["FE"]["elements"][2]['A'] == 100, "Non-target modified!"
    print("✓ PASSED: Multiple targets work\n")


def test_sampling():
    """Test automatic sampling"""
    print("="*70)
    print("TEST 6: Automatic sampling")
    print("="*70)
    
    config = {
        'E': 200e3,
        'load': 1000
    }
    
    E_var = Variable(name="E", kind="uniform", params={"low": 190e3, "high": 210e3})
    E_var.add_target(doc="sim", path="E")
    
    load_var = Variable(name="F", kind="uniform", params={"low": 500, "high": 1500})
    load_var.add_target(doc="sim", path="load")
    
    var_set = VariableSet(variables=[E_var, load_var])
    rng = np.random.default_rng(42)
    
    sampled_configs, samples = var_set.sample_configs(
        {"sim": config},
        n_samples=5,
        rng=rng
    )
    
    assert len(sampled_configs) == 5, "Wrong number of configs!"
    assert samples.shape == (5, 2), "Wrong sample array shape!"
    
    for i, cfg_dict in enumerate(sampled_configs):
        assert cfg_dict["sim"]['E'] == samples[i, 0], f"Sample {i} E mismatch!"
        assert cfg_dict["sim"]['load'] == samples[i, 1], f"Sample {i} load mismatch!"
    
    assert config['E'] == 200e3, "Original config modified!"
    print("✓ PASSED: Automatic sampling works\n")


def test_single_config_helper():
    """Test inject_single_config helper"""
    print("="*70)
    print("TEST 7: Single config helper function")
    print("="*70)
    
    config = {
        'param1': 10,
        'param2': 20
    }
    
    var1 = Variable(name="p1", kind="fixed", params={"value": 15})
    var1.add_target(doc="any", path="param1")
    
    var2 = Variable(name="p2", kind="fixed", params={"value": 25})
    var2.add_target(doc="any", path="param2")
    
    new_config = inject_single_config(config, [var1, var2], {"p1": 15, "p2": 25})
    
    assert new_config['param1'] == 15, "param1 not updated!"
    assert new_config['param2'] == 25, "param2 not updated!"
    assert config['param1'] == 10, "Original modified!"
    print("✓ PASSED: Single config helper works\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Testing Variable.targets functionality")
    print("#"*70 + "\n")
    
    test_basic_injection()
    test_indexed_path()
    test_nested_path()
    test_wildcard_path()
    test_multiple_targets()
    test_sampling()
    test_single_config_helper()
    
    print("#"*70)
    print("# ALL TESTS PASSED ✓")
    print("#"*70 + "\n")


if __name__ == "__main__":
    run_all_tests()

