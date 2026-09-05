"""Command-line entry point; supplied problem files remain read-only."""
from dataclasses import replace
from pathlib import Path
import argparse
import time
from .model import Site
from .optimize import SearchBudget,optimize_q2,optimize_q3
from .optics import evaluate
from .io import ROOT,safe_output,input_hashes,load_attachment,load_field,save_field,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='CUMCM 2023 A: interacting ray tracing and field optimization')
    parser.add_argument('stage',nargs='?',default='all',choices=['all','q1','q2','q3','verify'])
    parser.add_argument('--output',type=Path,default=ROOT/'output'/'validated')
    parser.add_argument('--profile',choices=['smoke','standard','research'],default='research')
    parser.add_argument('--seed',type=int,default=202309)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true',help='Reuse screening records; rescore all promoted designs')
    parser.add_argument('--rays',type=int,default=4096,help='Final Sobol rays per mirror and time')
    parser.add_argument('--replicates',type=int,default=4,help='Independent final scrambles')
    args=parser.parse_args(argv)
    safe_output(args.output/'run.json');args.output.mkdir(parents=True,exist_ok=True)
    before=input_hashes()
    if args.profile=='smoke':
        budget=SearchBudget(global_candidates=24,finalists=8,refinement_trials=4,q3_trials=6,
                            search_rays=64,refinement_rays=128)
    elif args.profile=='standard':budget=SearchBudget()
    else:
        budget=SearchBudget(global_candidates=512,finalists=48,refinement_trials=72,q3_trials=90,
                            search_rays=128,refinement_rays=256)
    budget=replace(budget,seed=args.seed,workers=args.workers)
    site=Site();start=time.time()
    try:
        if args.stage in ('all','q1'):
            field=load_attachment();save_field(field,args.output/'q1.npz')
            e=evaluate(field,site,rays=args.rays,seed=args.seed+10000)
            write_json({'power_MW':e.annual_power_kw/1000,'unit_power':e.unit_power,
                        'monthly':e.monthly(),'rays':args.rays},args.output/'q1_initial.json')
            print(f'Q1: {e.annual_power_kw/1000:.6f} MW, {e.unit_power:.6f} kW/m²',flush=True)
        if args.stage in ('all','q2'):optimize_q2(args.output,site,budget,resume=args.resume)
        if args.stage in ('all','q3'):
            optimize_q3(load_field(args.output/'q2.npz'),args.output,site,budget)
        if args.stage in ('all','verify'):
            from .verification import verify_all
            verify_all(args.output,site,budget,args.rays,args.replicates)
    finally:
        after=input_hashes()
        if before!=after:raise RuntimeError('Input hash mismatch: problem/ must remain unchanged')
        write_json({'stage':args.stage,'profile':args.profile,'budget':budget,'site':site,
                    'input_sha256':before,'input_unchanged':True,'seconds':time.time()-start},
                   args.output/f'run_{args.stage}.json')


if __name__=='__main__':main()
