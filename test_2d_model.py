import importlib.util
import math
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parent / "2d_model.py"
spec = importlib.util.spec_from_file_location("ising2d_model", MODULE_PATH)
ising2d_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ising2d_model)

enumeration_observables_2d = ising2d_model.enumeration_observables_2d
transfer_matrix_observables_2d = ising2d_model.transfer_matrix_observables_2d
trg_observables_2d = ising2d_model.trg_observables_2d


class TestIsing2DMethods(unittest.TestCase):

    def test_transfer_matches_enumeration(self) -> None:
        Lx, Ly = 3, 2
        T = 2.1
        J = 0.7
        h = 0.35

        enum_res = enumeration_observables_2d(Lx, Ly, T, J, h)
        tm_res = transfer_matrix_observables_2d(Lx, Ly, T, J, h, field_eps=1e-6)

        self.assertTrue(math.isfinite(enum_res.free_energy_per_spin))
        self.assertTrue(math.isfinite(tm_res.free_energy_per_spin))
        self.assertAlmostEqual(enum_res.free_energy_per_spin,
                               tm_res.free_energy_per_spin,
                               places=10)
        self.assertAlmostEqual(enum_res.susceptibility_per_spin,
                               tm_res.susceptibility_per_spin,
                               delta=5e-5)
        self.assertAlmostEqual(enum_res.heat_capacity_per_spin,
                               tm_res.heat_capacity_per_spin,
                               delta=5e-4)

    def test_trg_reasonable_accuracy(self) -> None:
        Lx = Ly = 4
        T = 2.3
        J = 1.0
        h = 0.0

        enum_res = enumeration_observables_2d(Lx, Ly, T, J, h)
        trg_res = trg_observables_2d(Lx, Ly, T, J, h, chi=32, field_eps=1e-6)

        self.assertTrue(math.isfinite(trg_res.free_energy_per_spin))
        self.assertLess(abs(enum_res.free_energy_per_spin - trg_res.free_energy_per_spin), 5e-4)
        self.assertLess(abs(enum_res.susceptibility_per_spin - trg_res.susceptibility_per_spin), 1e-2)
        self.assertTrue(math.isfinite(trg_res.heat_capacity_per_spin))


if __name__ == "__main__":
    unittest.main()
