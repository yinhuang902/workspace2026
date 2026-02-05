import pytest as pytest
from snoglode.utils.logging import IterLogger, MockIterLogger
import time
import numpy as np

def test_logger_rejects_invalid_log_level(tmp_path):
    with pytest.raises(AssertionError):
        IterLogger(tmp_path / "log", "VERBOSE")

def test_logger_creates_directory(tmp_path):
    log_path = tmp_path / "nested" / "run"
    logger = IterLogger(str(log_path), "INFO")
    assert log_path.with_suffix(".txt").exists()

def test_logger_writes_header(tmp_path):
    logger = IterLogger(str(tmp_path / "log"), "INFO")
    logger.logfile.close()
    content = (tmp_path / "log.txt").read_text()
    assert "SNoGloDe Log" in content

def test_init_timing_accumulates(tmp_path):
    logger = IterLogger(str(tmp_path / "log"), "INFO")
    logger.init_start()
    time.sleep(0.001)
    logger.init_stop()
    assert logger.total_init > 0

def test_update_increments_iter_and_resets(tmp_path):
    logger = IterLogger(str(tmp_path / "log"), "INFO")
    logger.alg_start(time.perf_counter())
    logger.lb_start()
    logger.lb_stop()
    logger.update()
    assert logger.iter == 1
    assert np.all(logger.lb == 0)

def test_update_writes_expected_sections(tmp_path):
    logger = IterLogger(str(tmp_path / "log"), "INFO")
    logger.alg_start(time.perf_counter())
    logger.update()
    logger.logfile.close()
    content = (tmp_path / "log.txt").read_text()
    assert "k,0" in content
    assert "LB," in content
    assert "CG," in content
    assert "UB," in content
    assert "Bound," in content
    assert "Branch," in content

def test_complete_writes_summary(tmp_path):
    logger = IterLogger(str(tmp_path / "log"), "INFO")
    logger.alg_start(time.perf_counter())
    logger.update()
    logger.complete()
    logger.logfile.close()
    content = (tmp_path / "log.txt").read_text()
    assert "SNoGloDe Summary" in content
    assert "Total time:" in content
    assert "Percent time spent" in content

def test_mock_logger_does_not_crash():
    logger = MockIterLogger()
    logger.alg_start(0)
    logger.init_start()
    logger.lb_start()
    logger.update()
    logger.complete()