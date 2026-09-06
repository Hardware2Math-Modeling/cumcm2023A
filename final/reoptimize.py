"""Resume the audited designs with new search, retaining reproducible snapshots.

Run before reproduce.py; final certification and assets are separate steps.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import shutil
from heliostat.io import load_field,save_field,write_json
from heliostat.model import Site
from heliostat.optimize import SearchBudget,tower_phase_search,height_profile_search,decode
from heliostat.refine import polish_q3

OUT=ROOT/'final/data';SNAP=ROOT/'final/audit/before_revision'


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('problem',choices=['q2','q3'])
    args=parser.parse_args();SNAP.mkdir(parents=True,exist_ok=True)
    for name in ('q2.npz','q2_search.json','q2_search_history.json','q3_search.json','q3_search_history.json'):
        if not (SNAP/name).exists():shutil.copy2(OUT/name,SNAP/name)
    if not (SNAP/'q3.npz').exists():shutil.copy2(ROOT/'final/audit/q3_baseline.npz',SNAP/'q3.npz')
    p=json.loads((SNAP/'q2_search.json').read_text())['parameters']
    budget=SearchBudget(target_kw=60200.,refinement_rays=512,search_rays=128,workers=2)
    if args.problem=='q2':
        # Retain the original Q2 reserve so its recovered incumbent is admissible.
        from dataclasses import replace
        budget=replace(budget,target_kw=60150.)
        parameters,field,e,history=tower_phase_search(p,Site(),budget)
        config=json.loads((SNAP/'q2_search.json').read_text())
        config.update(parameters=parameters,decoded=decode(parameters),power_kw=e.annual_power_kw,unit=e.unit_power,budget=budget,
                      revision='tower_phase_restart',previous_search='audit/before_revision/q2_search.json')
        save_field(field,OUT/'q2.npz');write_json(config,OUT/'q2_search.json')
        write_json(history,OUT/'q2_restart_history.json')
        write_json(history,OUT/'q2_search_history.json')
    else:
        field=load_field(SNAP/'q3.npz')
        field,e,profiles=height_profile_search(field,Site(),budget)
        write_json(profiles,OUT/'q3_profile_history.json');save_field(field,OUT/'q3_profile.npz')
        def checkpoint(f,h):
            save_field(f,OUT/'q3_revision_checkpoint.npz');write_json(h,OUT/'q3_polish_history.json')
        field,e,history=polish_q3(field,p,Site(),budget,checkpoint=checkpoint)
        field.name='q3';save_field(field,OUT/'q3.npz')
        write_json(dict(power_kw=e.annual_power_kw,unit=e.unit_power,budget=budget,
                        previous_search='audit/before_revision/q3_search.json',
                        profile_history='q3_profile_history.json',polish_history='q3_polish_history.json'),
                   OUT/'q3_revision.json')
        write_json(profiles+history,OUT/'q3_search_history.json')
        write_json(json.loads((OUT/'q3_revision.json').read_text()),OUT/'q3_search.json')
    print(args.problem,'completed',len(field),e.annual_power_kw,e.unit_power,flush=True)


if __name__=='__main__':main()
