import tempfile
import textwrap

from fmonz.config import load_config


def test_load_config_toml():
    # create a minimal configuration on the fly
    toml_str = textwrap.dedent(
        """
        [system]
        d = 2
        site_energies = [1.0, 2.0]
        couplings = [[0.0, 0.5], [0.5, 0.0]]

        [bath]
        temperature = 300.0
        reorganization_energy = 10.0
        cutoff = 50.0
        hierarchy_depth = 1

        [time]
        dt = 0.1
        n_steps = 5
        """
    )
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
        f.write(toml_str)
        fname = f.name
    cfg = load_config(fname)
    assert cfg.system.d == 2
    assert cfg.bath.temperature == 300.0
    assert cfg.time.n_steps == 5
