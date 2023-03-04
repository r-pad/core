import multiprocessing

import numpy as np

from rpad.core.distributed import NPSeed, distributed_eval


def __return_seeds(x: int, seed: NPSeed):
    return x**2


def test_distributed_eval():
    multiprocessing.set_start_method("spawn", force=True)
    args = [{"x": i} for i in range(10)]

    results, completeds = distributed_eval(
        __return_seeds,
        args,
        n_workers=2,
        n_proc_per_worker=1,
    )

    assert all(completeds)

    assert np.array_equal(results, [i**2 for i in range(10)])
