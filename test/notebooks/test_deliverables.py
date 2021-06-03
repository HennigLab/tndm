import runpy
import pytest


@pytest.fixture(scope='function')
def lorenz_generator_filename():
    return './test/notebooks/deliverables/lorenz_generator.py'

@pytest.mark.notebook
def test_smoke_lorenz_generator(lorenz_generator_filename):
    runpy.run_path(lorenz_generator_filename)
    assert True