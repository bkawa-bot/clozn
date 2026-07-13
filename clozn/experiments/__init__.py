"""Experiments package: ONE primitive over clozn's six run-scoped "hold everything constant, change
one thing, compare, with a receipt" operations (replay / counterfactual / receipt / branch /
swap_receipt). See experiment.py for the registry + dispatcher + envelope.
"""
from .experiment import REGISTRY, catalog, run_experiment, substrate_ok

__all__ = ["REGISTRY", "catalog", "run_experiment", "substrate_ok"]
