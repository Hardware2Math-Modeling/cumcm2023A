"""Check page budget, structure, numeric identities and unchanged source layouts.

Run with a Python environment containing pypdf. No ray tracing or optimization.
"""
from pathlib import Path
import csv,hashlib,json,math,re,statistics
from collections import defaultdict
from pypdf import PdfReader,PdfWriter

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def geometry(design,tower):
    bins=defaultdict(list);minimum=math.inf;margin=math.inf
    for i,a in enumerate(design):
        x,y,w,h,z=a;cell=(math.floor(x/32),math.floor(y/32))
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in bins[(cell[0]+dx,cell[1]+dy)]:
                    b=design[j];dist=math.hypot(x-b[0],y-b[1])
                    minimum=min(minimum,dist);margin=min(margin,dist-max(w,b[2])-5)
        bins[cell].append(i)
    # A violation lies within 13 m, so the 32 m cells include every violating pair.
    assert minimum<32 and margin>0
    ground=min(a[4]-a[3]/2 for a in design)
    site=min(350-math.hypot(a[0],a[1]) for a in design)
    exclusion=min(math.hypot(a[0]-tower[0],a[1]-tower[1])-100 for a in design)
    assert ground>0 and site>=-1e-8 and exclusion>=-1e-8
    assert all(2<=a[3]<=a[2]<=8 and 2<=a[4]<=6 for a in design)
    return dict(minimum_spacing_m=minimum,spacing_surplus_m=margin,ground_margin_m=ground,
                center_boundary_margin_m=site,center_exclusion_margin_m=exclusion)

def main():
    source=json.loads((HERE/'data/paper_sources.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in source['inputs'].items())
    original=json.loads((ROOT/'final/info_figures/manifest.json').read_text())['input_sha256']
    assert all(sha(ROOT/p)==h for p,h in original.items())
    expected=json.loads((ROOT/'info/data/expected.json').read_text())
    metrics={}
    for q in (1,2,3):
        p=HERE/f'data/q{q}.txt';assert sha(p)==sha(ROOT/f'info/data/q{q}.txt')
        lines=p.read_text().splitlines();head=lines[0].split();tower=list(map(float,head[1:3]))
        a=[list(map(float,line.split())) for line in lines[1:] if line.strip()]
        assert len(a)==int(head[0])==expected[str(q)]['N']
        area=sum(v[2]*v[3] for v in a);t=rows(HERE/f'data/q{q}_times.csv')
        per=rows(HERE/f'data/q{q}_times.csv.mirrors.csv')
        assert len(t)==60 and len(per)==len(a)
        assert [int(r['id']) for r in per]==list(range(1,len(a)+1))
        for r in t:
            assert abs(r['power_MW']*1000/area-r['unit_kW_m2'])<1e-10
            assert abs(r['dni']*area*r['optical']/1000-r['power_MW'])<1e-7
            assert all(0<=r[k]<=1 for k in ('optical','cosine','shadow_block','intercept','atmospheric'))
        power=statistics.mean(r['power_MW'] for r in t)
        assert abs(sum(v[2]*v[3]*r['unit_kW_m2'] for v,r in zip(a,per))/1000-power)<1e-7
        assert abs(power-expected[str(q)]['power_MW'])<1e-7
        assert abs(area-expected[str(q)]['area_m2'])<1e-6
        metrics[q]=dict(count=len(a),area_m2=area,power_MW=power,geometry=geometry(a,tower))
    stats=json.loads((HERE/'data/verification_summary.json').read_text())
    for q in ('2','3'):
        d=stats['independent'][q];x=d['replicates_MW']
        lower=statistics.mean(x)-2.3533634348018264*statistics.stdev(x)/2
        assert abs(lower-d['lower95_MW'])<1e-10 and lower>60
    aux=(HERE/'paper.aux').read_text()
    body=int(re.search(r'\\newlabel\{main-end\}\{\{8\}\{(\d+)\}',aux).group(1))
    pdf=PdfReader(HERE/'paper.pdf');main_pages=body+1
    assert 25<body and main_pages<=30,(body,main_pages)
    assert '关键词' in pdf.pages[0].extract_text()
    assert '参考文献' in pdf.pages[main_pages].extract_text()
    text='\n'.join(p.extract_text() for p in pdf.pages)
    for title in ('1.1 问题背景','1.2 问题重述','5.1.1','5.2.1','5.3.1','6 模型检验与结果分析','7 模型评价与推广改进','8 结论','人工智能工具使用说明'):
        assert title in text,title
    assert '??' not in text and '\ufffd' not in text
    log=(HERE/'paper.log').read_text()
    assert not re.search(r'Overfull|LaTeX Warning:|Package .* Warning:|! .*Error',log)
    for label,indices in [('paper_main',range(main_pages)),('appendices',range(main_pages,len(pdf.pages)))]:
        out=PdfWriter()
        for i in indices:out.add_page(pdf.pages[i])
        out.add_metadata({'/Title':'定日镜场优化设计：'+('摘要与正文' if label=='paper_main' else '附录')})
        with (HERE/(label+'.pdf')).open('wb') as f:out.write(f)
    report=dict(abstract_pages=1,body_pages=body,pages_excluding_appendices=main_pages,
                appendix_pages=len(pdf.pages)-main_pages,total_pages=len(pdf.pages),page_limit_passed=True,
                structure_passed=True,latex_warnings=0,source_inputs_unchanged=True,
                metrics=metrics,independent_capacity_check_passed=True,pdf_sha256=sha(HERE/'paper.pdf'))
    (HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
