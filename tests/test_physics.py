import unittest
from dataclasses import replace
import numpy as np
from heliostat.model import Field,Site,solar_vector,time_grid,validate_field
from heliostat.geometry import cylinder_intersection,rectangle_hits
from heliostat.optics import evaluate
from heliostat.io import safe_output,PROBLEM


class PhysicsTests(unittest.TestCase):
    def test_solar_east_and_symmetry(self):
        a,b=solar_vector(0,9),solar_vector(0,15)
        self.assertGreater(a[0],0)
        np.testing.assert_allclose(a,[-b[0],b[1],b[2]],atol=1e-14)
        _,s,_=time_grid()
        np.testing.assert_allclose(np.linalg.norm(s,axis=1),1,atol=1e-14)

    def test_specular_center_and_horizontal_edges(self):
        f=Field([[0,150,4],[100,-130,5]],6,6,[0,0])
        sun=solar_vector(0,9)
        t,n,u,v,d=f.geometry(sun)
        np.testing.assert_allclose(2*(n@sun)[:,None]*n-sun,t,atol=1e-14)
        np.testing.assert_allclose(u[:,2],0,atol=1e-14)
        np.testing.assert_allclose(np.cross(u,v),n,atol=1e-14)

    def test_cylinder_first_surface(self):
        p=np.array([[10,0,80],[0,0,70],[10,4,80],[10,0,80]])
        d=np.array([[-1,0,0],[0,0,1],[-1,0,0],[1,0,0]])
        t,k=cylinder_intersection(p,d,[0,0],3.5,76,84)
        np.testing.assert_allclose(t[:2],[6.5,6])
        np.testing.assert_array_equal(k,[1,2,0,0])
        self.assertTrue(np.all(np.isinf(t[2:])))

    def test_rectangle_edges_and_backward_rays(self):
        p=np.array([[0,0,0],[1.1,0,0],[0,0,2]])
        d=np.tile([0,0,1],(3,1))
        hit=rectangle_hits(p,d,np.array([0,0,1]),np.array([0,0,1]),
                           np.array([1,0,0]),np.array([0,1,0]),2,2)
        np.testing.assert_array_equal(hit,[True,False,False])

    def test_independent_native_reference_and_culling(self):
        # Deliberately overlapping/unequal mirrors exercise shadow unions and blocking.
        rng=np.random.default_rng(44)
        centers=np.c_[rng.uniform(-35,35,(24,2)),rng.uniform(3.5,6,24)]
        centers[:,1]+=160
        f=Field(centers,rng.uniform(4,8,24),rng.uniform(3,6,24),[0,-70])
        site=replace(Site(),slope_error=.001)
        args=dict(site=site,rays=128,seed=918,time_indices=[0,2,27,59])
        fast=evaluate(f,backend='native',**args)
        slow=evaluate(f,backend='numpy',all_pairs=True,**args)
        np.testing.assert_allclose(fast.metrics,slow.metrics,atol=2e-12)
        self.assertLess(float(fast.metrics[:,:,1].min()),.9)
        self.assertTrue(np.all(fast.power_kw>=0))
        self.assertTrue(np.all(fast.metrics[:,:,6]<=fast.metrics[:,:,0]+1e-12))
        np.testing.assert_allclose(fast.metrics[:,:,0]*fast.metrics[:,:,1]*fast.metrics[:,:,2],fast.metrics[:,:,6],atol=1e-14)

    def test_receiver_size_and_solar_cone_matter(self):
        f=Field([[0,330,5]],8,8,[0,0])
        small=evaluate(f,replace(Site(),receiver_radius=1),rays=2048,time_indices=[2])
        large=evaluate(f,replace(Site(),receiver_radius=20,receiver_height=40),rays=2048,time_indices=[2])
        self.assertGreater(large.annual_power_kw,small.annual_power_kw*2)
        self.assertAlmostEqual(float(large.metrics[0,0,2]),1,places=12)
        pencil=evaluate(f,replace(Site(),solar_radius=0),rays=2048,time_indices=[2])
        spread=evaluate(f,replace(Site(),solar_radius=.015),rays=2048,time_indices=[2])
        self.assertGreater(pencil.annual_power_kw,spread.annual_power_kw)

    def test_permutation_invariance(self):
        f=Field([[0,130,4],[0,142,4],[12,130,5]],6,6,[0,0])
        a=evaluate(f,rays=128,time_indices=[0,29])
        b=evaluate(f.subset([2,0,1]),rays=128,time_indices=[0,29])
        np.testing.assert_allclose(a.power_kw[:,[2,0,1]],b.power_kw,atol=1e-12)

    def test_constraints_and_input_protection(self):
        f=Field([[0,150,4],[10,150,4]],6,6,[0,0])
        self.assertIn('pair spacing',validate_field(f)['violations'])
        f=Field([[0,150,3]],6,6,[0,0])
        self.assertIn('ground clearance during full rotation',validate_field(f)['violations'])
        with self.assertRaises(ValueError):safe_output(PROBLEM/'result2.xlsx')

    def test_area_weighted_energy_identity(self):
        f=Field([[0,150,4],[100,150,5]],np.array([4,8]),np.array([4,6]),[0,0])
        e=evaluate(f,rays=128,time_indices=[0,29])
        _,_,dni=time_grid()
        np.testing.assert_allclose(e.rows()[:,0]*dni[e.time_indices]*f.area.sum(),e.power_kw.sum(axis=1),atol=1e-10)


if __name__=='__main__':unittest.main()
