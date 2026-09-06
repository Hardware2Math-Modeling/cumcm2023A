"""Check the numerical provenance and table/array consistency of the paper."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import csv
import json
import hashlib
import numpy as np
from heliostat.io import load_field,input_hashes
from heliostat.model import validate_field,time_grid


def main():
    data=HERE/'data'
    audit=json.loads((data/'verification.json').read_text())
    assert audit['input_sha256']==input_hashes(),'Problem inputs changed'
    for path,expected in audit['source_sha256'].items():
        actual=hashlib.sha256((HERE.parent/path).read_bytes()).hexdigest()
        assert actual==expected,f'Computational source changed: {path}'
    _,_,dni=time_grid()
    report={}
    for key in ('q1','q2','q3'):
        assert hashlib.sha256((data/f'{key}.npz').read_bytes()).hexdigest()==audit[key]['design_sha256'],f'Stale verification for {key}'
        f=load_field(data/f'{key}.npz')
        z=np.load(data/f'{key}_optics.npz')
        check=validate_field(f,uniform=key=='q2',whole_mirror=key!='q1')
        assert check['valid'],check
        assert z['metrics'].shape==(60,len(f),7)
        assert np.all(z['power_kw']>=0)
        np.testing.assert_allclose(z['area'],f.area,atol=1e-12)
        np.testing.assert_allclose(z['metrics'][:,:,0]*z['metrics'][:,:,1]*z['metrics'][:,:,2],z['metrics'][:,:,6],atol=2e-12)
        p=dni[:,None]*f.area[None,:]*z['atmospheric'][None,:]*.92*z['metrics'][:,:,6]
        np.testing.assert_allclose(p,z['power_kw'],atol=1e-9)
        np.testing.assert_allclose(np.array(audit[key]['monthly']).mean(axis=0),audit[key]['annual'],atol=1e-10)
        np.testing.assert_allclose(p.sum(axis=1).reshape(12,5).mean(axis=1)/1000,np.array(audit[key]['monthly'])[:,5],atol=1e-9)
        assert abs(p.sum(axis=1).mean()/f.area.sum()-audit[key]['unit_power_kW_m2'])<1e-10
        if key!='q1':
            assert audit[key]['uncertainty']['lower_95_one_sided_kw']>=60000
            with (data/f'result{key[-1]}.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.reader(stream))
            assert len(rows)==len(f)+1
            v=np.array(rows[1:],float)
            np.testing.assert_allclose(v[:,5:8],f.centers,atol=1e-12)
            np.testing.assert_allclose(v[:,3],f.widths,atol=1e-12)
            np.testing.assert_allclose(v[:,4],f.heights,atol=1e-12)
        report[key]=dict(count=len(f),area=float(f.area.sum()),power_MW=audit[key]['power_MW'],valid=True)
    assert audit['q3_improves_q2']
    assert audit['paired_q3_minus_q2']['lower_95_one_sided_kw']>0
    if 'revision_q3_comparison' in audit:
        assert audit['revision_q3_comparison']['paired_unit_gain']['lower_95_one_sided_kw']>0
    result=dict(checks='passed',input_unchanged=True,computational_source_unchanged=True,designs=report)
    (data/'paper_consistency.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
