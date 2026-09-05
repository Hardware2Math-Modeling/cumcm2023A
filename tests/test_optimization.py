import unittest
from dataclasses import replace
import numpy as np
from heliostat.model import Field,Site,validate_field
from heliostat.optics import evaluate
from heliostat.optimize import candidates,encode,trim_field
from heliostat.verification import aggregate,uncertainty


class OptimizationTests(unittest.TestCase):
    def test_layout_generators_enforce_all_pairs(self):
        for family in ('staggered','rings','adaptive_rings'):
            for w in (4.5,6.5,8):
                p=encode((31,-97),w,w,z=w/2+.2,family=family)
                p[7]=.71;p[8]=.19;p[9]=.83
                f=candidates(p)
                check=validate_field(f,uniform=True,whole_mirror=True)
                self.assertTrue(check['valid'],check)

    def test_removing_occluders_cannot_reduce_survivor_power(self):
        f=Field([[0,200,4],[0,208,4],[8,200,4],[8,208,4]],6,6,[0,0])
        full=evaluate(f,rays=256,time_indices=[0,2,27,59])
        selected=evaluate(f.subset([0,3]),rays=256,time_indices=[0,2,27,59])
        self.assertTrue(np.all(selected.power_kw>=full.power_kw[:,[0,3]]-1e-12))

    def test_no_mirror_interaction_is_an_upper_bound(self):
        f=Field([[0,200,4],[0,208,4],[8,200,4],[8,208,4]],6,6,[0,0])
        a=evaluate(f,rays=256,time_indices=[0,2,27,59])
        b=evaluate(f,rays=256,time_indices=[0,2,27,59],interactions=False)
        self.assertTrue(np.all(b.power_kw>=a.power_kw-1e-12))

    def test_energy_aggregation_preserves_factorization(self):
        f=Field([[0,200,4],[0,208,5],[8,200,4]],np.array([6,7,8]),5,[0,0])
        values=[evaluate(f,rays=64,seed=s,time_indices=[0,2]) for s in (13,19,29)]
        mean=aggregate(values)
        np.testing.assert_allclose(mean.metrics[:,:,0]*mean.metrics[:,:,1]*mean.metrics[:,:,2],mean.metrics[:,:,6],atol=1e-14)
        np.testing.assert_allclose(mean.power_kw,np.mean([e.power_kw for e in values],axis=0))

    def test_exhaustive_far_field_culling(self):
        rng=np.random.default_rng(171)
        xy=rng.uniform([-330,-250],[330,340],(55,2))
        f=Field(np.c_[xy,rng.uniform(3.1,6,55)],8,6,[0,-235])
        site=replace(Site(),slope_error=.002)
        a=evaluate(f,site,rays=128,time_indices=[0,4,55,59])
        b=evaluate(f,site,rays=128,time_indices=[0,4,55,59],all_pairs=True)
        np.testing.assert_allclose(a.metrics,b.metrics,atol=1e-14)

    def test_uncertainty_requires_independent_replicates(self):
        with self.assertRaises(ValueError):uncertainty([60001])
        stats=uncertainty([59980,60000,60020,60010])
        self.assertLess(stats['lower_95_one_sided_kw'],stats['mean_kw'])


if __name__=='__main__':unittest.main()
