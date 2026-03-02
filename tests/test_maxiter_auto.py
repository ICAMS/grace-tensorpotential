import pytest
from tensorpotential.cli.gracemaker import compute_maxiter

def test_compute_maxiter_adam():
    # 50k updates / 100 batches = 500 epochs
    assert compute_maxiter(50_000, 100, "Adam") == 500
    # 50k updates / 77 batches = 649.35... -> rounds to 650
    assert compute_maxiter(50_000, 77, "Adam") == 650
    # Clamping
    assert compute_maxiter(100_000, 2, "Adam") == 5000 # clamped [10, 5000]
    assert compute_maxiter(10, 100, "Adam") == 10 # clamped min

def test_compute_maxiter_bfgs():
    # 1 update = 1 epoch
    assert compute_maxiter(500, 100, "BFGS") == 500
    assert compute_maxiter(100, 10, "L-BFGS-B") == 100
    # Rounding
    assert compute_maxiter(123, 10, "BFGS") == 120
    # Clamping
    assert compute_maxiter(10000, 10, "BFGS") == 5000
    assert compute_maxiter(5, 10, "BFGS") == 10

if __name__ == "__main__":
    pytest.main([__file__])
