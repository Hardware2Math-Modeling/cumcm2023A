"""Units throughout: metres, radians, kW, kW/m²; coordinates east/north/up."""
from dataclasses import dataclass, replace
from datetime import date
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Site:
    latitude_deg: float = 39.4
    altitude_km: float = 3.0
    field_radius: float = 350.0
    exclusion_radius: float = 100.0
    tower_height: float = 80.0  # receiver CENTRE, not its bottom
    receiver_radius: float = 3.5
    receiver_height: float = 8.0
    reflectivity: float = 0.92
    solar_radius: float = 0.00465  # apparent solar angular radius, rad
    tower_radius: float = 2.0  # unspecified shaft: explicit, configurable assumption
    slope_error: float = 0.0  # baseline ideal flat mirrors; sensitivity in radians


def solar_vector(day, solar_time, latitude_deg=39.4):
    phi = np.deg2rad(latitude_deg)
    delta = np.arcsin(np.sin(2*np.pi*day/365)*np.sin(np.deg2rad(23.45)))
    omega = np.pi/12*(solar_time-12)
    return np.array([-np.cos(delta)*np.sin(omega),
                     np.sin(delta)*np.cos(phi)-np.cos(delta)*np.sin(phi)*np.cos(omega),
                     np.sin(delta)*np.sin(phi)+np.cos(delta)*np.cos(phi)*np.cos(omega)])


def dni(sun_z, altitude_km=3.0):
    a = 0.4237-0.00821*(6-altitude_km)**2
    b = 0.5055+0.00595*(6.5-altitude_km)**2
    c = 0.2711+0.01858*(2.5-altitude_km)**2
    return np.where(np.asarray(sun_z) > 0,
                    1.366*(a+b*np.exp(-c/np.maximum(sun_z, 1e-12))), 0.)


def time_grid(site=Site()):
    days = [(date(2023,m,21)-date(2023,3,21)).days % 365 for m in range(1,13)]
    labels = [(m+1,t) for m in range(12) for t in (9.,10.5,12.,13.5,15.)]
    sun = np.array([solar_vector(days[m-1],t,site.latitude_deg) for m,t in labels])
    return labels, sun, dni(sun[:,2], site.altitude_km)


@dataclass
class Field:
    centers: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    tower: np.ndarray
    name: str = "field"

    def __post_init__(self):
        self.centers = np.ascontiguousarray(self.centers, dtype=float)
        n = len(self.centers)
        self.widths = np.broadcast_to(self.widths, (n,)).astype(float).copy()
        self.heights = np.broadcast_to(self.heights, (n,)).astype(float).copy()
        self.tower = np.asarray(self.tower, dtype=float).copy()
        if self.centers.shape != (n,3) or self.tower.shape != (2,) or n == 0:
            raise ValueError("Nonempty N×3 centers and 2D tower coordinates required")
        if not all(np.all(np.isfinite(a)) for a in
                   (self.centers,self.widths,self.heights,self.tower)):
            raise ValueError("Field contains nonfinite values")

    def __len__(self):
        return len(self.centers)

    @property
    def area(self):
        return self.widths*self.heights

    @property
    def radii(self):
        return np.hypot(self.widths,self.heights)/2

    def subset(self, indices, name=None):
        return Field(self.centers[indices],self.widths[indices],self.heights[indices],
                     self.tower, name or self.name)

    def copy(self, **kwargs):
        return replace(self, **kwargs)

    def geometry(self, sun, site=Site()):
        receiver = np.r_[self.tower,site.tower_height]
        target = receiver-self.centers
        distance = np.linalg.norm(target,axis=1)
        target /= distance[:,None]
        normal = target+sun
        normal /= np.linalg.norm(normal,axis=1)[:,None]
        u = np.c_[-normal[:,1],normal[:,0],np.zeros(len(self))]
        scale = np.linalg.norm(u,axis=1)
        vertical = scale < 1e-12
        u[vertical] = [1,0,0]
        u /= np.where(vertical,1,scale)[:,None]
        v = np.cross(normal,u)
        return target, normal, u, v, distance


def validate_field(field, site=Site(), uniform=False, whole_mirror=False):
    """Check all pairs conservatively: spacing >= max(width_i,width_j)+5.

    whole_mirror additionally keeps the circumscribed horizontal footprint inside
    the site/outside the exclusion disk. The attachment is never relocated.
    """
    c,w,h = field.centers,field.widths,field.heights
    footprint = field.radii if whole_mirror else np.zeros(len(field))
    site_margin = site.field_radius-np.linalg.norm(c[:,:2],axis=1)-footprint
    exclusion_margin = np.linalg.norm(c[:,:2]-field.tower,axis=1)-site.exclusion_radius-footprint
    pairs = cKDTree(c[:,:2]).query_pairs(float(w.max()+5+1e-6), output_type='ndarray')
    if len(pairs):
        separation = np.linalg.norm(c[pairs[:,0],:2]-c[pairs[:,1],:2],axis=1)
        margin = separation-np.maximum(w[pairs[:,0]],w[pairs[:,1]])-5
        pair_margin = float(margin.min())
    else:
        d,j = cKDTree(c[:,:2]).query(c[:,:2],k=min(2,len(field)))
        pair_margin = float(np.min(d[:,1]-np.maximum(w,w[j[:,1]])-5)) if len(field)>1 else 1e30
    violations = []
    for condition, message in [
        (np.all((w>=2-1e-9)&(w<=8+1e-9)&(h>=2-1e-9)&(h<=w+1e-9)),"mirror dimensions"),
        (np.all((c[:,2]>=2-1e-9)&(c[:,2]<=6+1e-9)),"installation height"),
        (np.all(c[:,2]-h/2>0),"ground clearance during full rotation"),
        (site_margin.min()>=-1e-8,"site boundary"),
        (exclusion_margin.min()>=-1e-8,"tower exclusion"),
        (pair_margin>=-1e-8,"pair spacing"),
        (np.linalg.norm(field.tower)<=site.field_radius,"tower outside site"),
        (not uniform or (np.ptp(w)<1e-9 and np.ptp(h)<1e-9 and np.ptp(c[:,2])<1e-9),"nonuniform Q2"),
    ]:
        if not condition:
            violations.append(message)
    return {"valid":not violations,"violations":violations,"count":len(field),
            "total_area_m2":float(field.area.sum()),"minimum_spacing_margin_m":pair_margin,
            "minimum_ground_clearance_m":float(np.min(c[:,2]-h/2)),
            "minimum_site_margin_m":float(site_margin.min()),
            "minimum_exclusion_margin_m":float(exclusion_margin.min()),
            "whole_mirror_footprint":whole_mirror}
