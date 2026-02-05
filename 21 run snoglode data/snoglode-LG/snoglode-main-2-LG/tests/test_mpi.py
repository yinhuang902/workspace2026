import pytest as pytest
import sys
import snoglode.utils.MPI as MPI

def test_serial_mpi_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)

    import importlib
    mpi_module = importlib.reload(MPI)

    assert mpi_module._haveMPI is False
    assert mpi_module.COMM_WORLD.rank == 0
    assert mpi_module.COMM_WORLD.size == 1

def test_mock_allreduce_identity():
    x = [1, 2, 3]
    y = MPI.COMM_WORLD.allreduce(x)
    assert y == x
    assert y is not x  # deepcopy check

def test_mock_bcast():
    data = {"a": 1}
    assert MPI.COMM_WORLD.bcast(data) == data

def test_mock_comm_api():
    assert MPI.COMM_WORLD.Get_rank() == 0
    assert MPI.COMM_WORLD.Get_size() == 1