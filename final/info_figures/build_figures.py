"""Paper figures from info's delivered layouts and unmodified C++ ray tracer.

Run with the project's scientific Python environment. Input geometry is never
edited; compact plots enlarge only the projected rectangle glyphs.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INFO = ROOT / 'info'
DATA = HERE / 'data'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build' / 'info_figure_matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages

SEED = 20260905
RAYS = 16384
SCALE = 1.20
COLORS = ['#54788c', '#237f83', '#c38239']
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal_teal', ['#deedef', '#91ccca', '#45aaa9', '#28758a', '#193d59'])
NAMES = {1: '问题一', 2: '问题二', 3: '问题三'}
OUTPUTS = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design(q):
    path = INFO / 'data' / f'q{q}.txt'
    header = path.read_text().splitlines()[0].split()
    a = np.loadtxt(path, skiprows=1)
    assert a.shape == (int(header[0]), 5)
    return a, np.array(header[1:3], float)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare_data():
    DATA.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'build' / 'info_figure_trace'
    cache.mkdir(parents=True, exist_ok=True)
    exe = cache / ('trace_' + digest(INFO / 'trace.cpp')[:16])
    if not exe.exists():
        subprocess.run(['c++', '-O3', '-std=c++17', str(INFO / 'trace.cpp'), '-o', str(exe)], check=True)
    expected = json.loads((INFO / 'data' / 'expected.json').read_text())

    def one(q):
        start = time.monotonic()
        source = INFO / 'data' / f'q{q}.txt'
        out = DATA / f'q{q}_times.csv'
        manifest = DATA / f'q{q}_trace.json'
        key = dict(input_sha256=digest(source), source_sha256=digest(INFO / 'trace.cpp'),
                   rays=RAYS, seed=SEED, solar_half_angle_rad=.00465, mast_radius_m=3.5)
        cached = manifest.exists() and out.exists() and Path(str(out) + '.mirrors.csv').exists()
        if cached:
            previous = json.loads(manifest.read_text())
            cached = (previous.get('settings') == key
                      and previous.get('time_csv_sha256') == digest(out)
                      and previous.get('per_mirror_csv_sha256') == digest(Path(str(out) + '.mirrors.csv')))
        if not cached:
            print(f'Q{q}: recomputing 60 times x {RAYS} rays per mirror', flush=True)
            proc = subprocess.run([str(exe), str(source), str(out), str(RAYS), str(SEED), '.00465', '3.5', '1'],
                                  check=True, capture_output=True, text=True)
            print(proc.stdout.strip(), flush=True)
        times = np.genfromtxt(out, delimiter=',', names=True)
        per = np.genfromtxt(str(out) + '.mirrors.csv', delimiter=',', names=True)
        a, tower = load_design(q)
        assert len(times) == 60 and len(per) == len(a)
        assert np.array_equal(per['id'], np.arange(1, len(a) + 1))
        power = float(times['power_MW'].mean())
        difference = abs(power - expected[str(q)]['power_MW'])
        assert difference < 1e-7, (q, difference)
        area = a[:, 2] * a[:, 3]
        assert abs(np.dot(area, per['unit_kW_m2']) / 1000 - power) < 1e-7
        assert q == 1 or power >= 60
        record = dict(settings=key, power_MW=power,
                      unit_kW_m2=float(times['unit_kW_m2'].mean()),
                      count=len(a), area_m2=float(area.sum()),
                      difference_from_expected_MW=difference,
                      time_csv_sha256=digest(out),
                      per_mirror_csv_sha256=digest(Path(str(out) + '.mirrors.csv')))
        write_json(manifest, record)
        print(f'Q{q}: verified against info/data/expected.json ({time.monotonic()-start:.1f} s)', flush=True)
        return q, (a, tower, times, per, record)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return dict(pool.map(one, (1, 2, 3)))


def configure():
    plt.rcParams.update({
        'font.family': ['Songti SC', 'Arial', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'path', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'text.color': '#24313b', 'axes.labelcolor': '#24313b',
        'xtick.color': '#46535d', 'ytick.color': '#46535d',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'lines.linewidth': 1.7, 'legend.frameon': False,
    })


def projected_rectangles(a, tower, scale=1.):
    """True top-view corners at equinox noon; horizontal mirror edge is preserved.

    The info tracer uses n = unit(s + r), u = unit(z cross n), v = n cross u.
    Here u is horizontal, so its horizontal projection is perpendicular to v's.
    These polygons are rectangles, not screen-space scatter markers.
    """
    centers = a[:, [0, 1, 4]]
    target = np.r_[tower, 80.] - centers
    target /= np.linalg.norm(target, axis=1)[:, None]
    lat = np.deg2rad(39.4)
    sun = np.array([0., -np.sin(lat), np.cos(lat)])
    normal = target + sun
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    u = np.cross(np.array([0., 0., 1.]), normal)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(normal, u)
    du = u[:, :2] * (a[:, 2] * scale / 2)[:, None]
    dv = v[:, :2] * (a[:, 3] * scale / 2)[:, None]
    corners = np.stack([a[:, :2]-du-dv, a[:, :2]+du-dv,
                        a[:, :2]+du+dv, a[:, :2]-du+dv], axis=1)
    assert np.allclose(corners.mean(axis=1), a[:, :2])
    assert np.allclose(np.einsum('ij,ij->i', du, dv), 0, atol=1e-10)
    return corners


def field_axes(ax, tower, title=None):
    ax.add_patch(Circle((0, 0), 350, fill=False, color='#59656b', lw=.8, zorder=4))
    ax.add_patch(Circle(tower, 100, fill=False, color='#899298', lw=.8,
                        linestyle=(0, (4, 3)), zorder=4))
    ax.plot(*tower, marker='+', markersize=10, markeredgewidth=1.6,
            color='#293c47', linestyle='none', zorder=6)
    ax.text(tower[0], tower[1]-19, '吸收塔', ha='center', va='top', fontsize=9)
    ax.set(xlim=(-355, 355), ylim=(-355, 355), aspect='equal',
           xlabel='东向坐标 x / m', ylabel='北向坐标 y / m', title=title)
    ax.set_xticks([-300, -150, 0, 150, 300])
    ax.set_yticks([-300, -150, 0, 150, 300])
    ax.tick_params(pad=3)


def mirrors(ax, a, tower, values, norm, cmap=THERMAL_CMAP, scale=SCALE):
    collection = PolyCollection(projected_rectangles(a, tower, scale),
                                array=np.asarray(values), cmap=cmap, norm=norm,
                                edgecolors='none', antialiaseds=False, zorder=2)
    ax.add_collection(collection)
    return collection


def note(fig, scale=SCALE):
    text = ('矩形按 3 月 21 日 12:00 镜姿投影绘制，符号放大 1.2 倍；镜心位置不变。'
            if scale != 1 else '镜面按 3 月 21 日 12:00 的俯视投影等比例绘制。')
    fig.text(.5, .015, text, ha='center', fontsize=8, color='#65717a')


def save(fig, name, book=None):
    target = HERE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(target.with_suffix('.' + extension), dpi=360,
                    bbox_inches='tight', pad_inches=.035)
    if book is not None:
        book.savefig(fig, bbox_inches='tight', pad_inches=.035)
    OUTPUTS.append(name)
    plt.close(fig)


def layout_figure(q, entry, norm, scale=SCALE):
    a, tower, times, per, record = entry
    fig = plt.figure(figsize=(7.0, 6.45))
    ax = fig.add_axes([.10, .115, .735, .805])
    im = mirrors(ax, a, tower, per['unit_kW_m2'], norm, scale=scale)
    field_axes(ax, tower)
    ax.set_title(f'{NAMES[q]}定日镜场布局', y=1.070, pad=0)
    cax = fig.add_axes([.87, .22, .023, .59])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('单位镜面面积年均热功率 / (kW/m$^2$)', labelpad=9)
    ax.text(.5, 1.025, f'{len(a):,} 面  |  {record["power_MW"]:.4f} MW  |  '
            f'{record["unit_kW_m2"]:.5f} kW/m$^2$', transform=ax.transAxes,
            ha='center', fontsize=9, color='#52616a')
    note(fig, scale)
    return fig


def comparison_figure(entries, norm, scale=SCALE):
    fig = plt.figure(figsize=(11.9, 5.45))
    for left, q, tag in ((.055, 2, '(a)'), (.505, 3, '(b)')):
        ax = fig.add_axes([left, .16, .395, .76])
        a, tower, _, per, record = entries[q]
        im = mirrors(ax, a, tower, per['unit_kW_m2'], norm, scale=scale)
        field_axes(ax, tower, f'{tag} {NAMES[q]}：{len(a):,} 面')
    cax = fig.add_axes([.925, .24, .012, .57])
    fig.colorbar(im, cax=cax, label='单位镜面面积年均热功率 / (kW/m$^2$)')
    note(fig, scale)
    return fig


def heterogeneity_figure(entry, scale=SCALE):
    a, tower, *_ = entry
    fig = plt.figure(figsize=(12.1, 5.5))
    for left, values, title, label, cmap, norm in (
        (.055, a[:, 2]*a[:, 3], '(a) 镜面面积', '镜面面积 / m$^2$', 'viridis_r', Normalize(11.5, 33.0625)),
        (.55, a[:, 4], '(b) 安装高度', '安装高度 / m', 'cividis', Normalize(2.925, 6.))):
        ax = fig.add_axes([left, .16, .355, .76])
        im = mirrors(ax, a, tower, values, norm, cmap, scale)
        field_axes(ax, tower, title)
        cax = fig.add_axes([left+.365, .25, .012, .54])
        fig.colorbar(im, cax=cax, label=label)
    note(fig, scale)
    return fig


def detail_figure(entry):
    a, tower, _, per, _ = entry
    norm = Normalize(.15, .75)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.90, wspace=.24)
    for ax, limits, title in zip(axs, [(-60, 60, 90, 180), (-60, 60, 245, 335)],
                                ['(a) 中部交错环带', '(b) 北部外环带']):
        mirrors(ax, a, tower, per['unit_kW_m2'], norm, scale=1.)
        x0, x1, y0, y1 = limits
        ax.set(xlim=(x0,x1), ylim=(y0,y1), aspect='equal', title=title,
               xlabel='东向坐标 x / m', ylabel='北向坐标 y / m')
        ax.grid(color='#dfe5e8', linewidth=.5, alpha=.7, zorder=0)
    fig.text(.5, .035, '第三问局部放大：镜面使用真实尺寸与参考时刻投影，可见交错环距和不同矩形尺寸。',
             ha='center', fontsize=9, color='#65717a')
    return fig


def radial_figure(entry):
    a, tower, *_ = entry
    radius = np.linalg.norm(a[:, :2]-tower, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig.subplots_adjust(left=.065, right=.98, bottom=.18, top=.88, wspace=.25)
    sizes, counts = np.unique(a[:, 3], return_counts=True)
    bars = axs[0].bar(np.arange(len(sizes)), counts, width=.66, color=COLORS[0])
    axs[0].set(xticks=np.arange(len(sizes)), xticklabels=[f'{v:g}' for v in sizes],
               xlabel='镜面高度 / m（宽度均为 5.75 m）', ylabel='镜数 / 面',
               title='(a) 第三问镜面规格分布', ylim=(0, 3150))
    axs[0].bar_label(bars, padding=3, fontsize=9)
    edges = np.linspace(radius.min()-1e-7, radius.max()+1e-7, 9)
    band = np.clip(np.searchsorted(edges, radius, side='right')-1, 0, 7)
    levels = np.array([np.unique(a[band == i, 4]).item() for i in range(8)])
    axs[1].stairs(levels, edges, color=COLORS[1], linewidth=2.1, baseline=None)
    axs[1].plot((edges[:-1]+edges[1:])/2, levels, marker='s', linestyle='none',
                color=COLORS[1], markersize=4)
    axs[1].set(xlabel='距吸收塔半径 / m', ylabel='安装高度 / m',
               title='(b) 八个径向区间的安装高度', xlim=(95, 415), ylim=(2.65, 6.32))
    for ax in axs:
        ax.grid(axis='y', alpha=.2, linewidth=.6)
        ax.set_axisbelow(True)
    return fig


def monthly_figures(entries, book):
    months = np.arange(1, 13)
    monthly = {q: {key: np.array([e[2][key][e[2]['month']==m].mean() for m in months])
                    for key in e[2].dtype.names if key not in ('month', 'time')}
               for q, e in entries.items()}
    fig, axs = plt.subplots(1, 2, figsize=(10.7, 3.95))
    fig.subplots_adjust(left=.065, right=.98, bottom=.17, top=.88, wspace=.25)
    for q in (1,2,3):
        axs[0].plot(months, monthly[q]['power_MW'], label=NAMES[q], color=COLORS[q-1],
                    marker='s' if q==2 else '^', markersize=4,
                    linestyle='--' if q==2 else '-', zorder=5 if q==2 else 3)
    axs[0].axhline(60, color='#90989d', linestyle=(0,(4,3)), linewidth=.8)
    axs[0].text(1.2,61.2,'年均额定值 60 MW', fontsize=8, color='#6c7880')
    axs[0].set(title='(a) 三问逐月平均热功率', xlabel='月份', ylabel='热功率 / MW')
    for q in (2,3):
        axs[1].plot(months, monthly[q]['unit_kW_m2'], marker='s' if q==2 else '^',
                    markersize=4, color=COLORS[q-1], label=NAMES[q])
    axs[1].set(title='(b) 第二、三问单位面积热功率', xlabel='月份', ylabel='热功率 / (kW/m$^2$)')
    for ax in axs:
        ax.set_xticks(months); ax.grid(alpha=.18); ax.legend(fontsize=9, loc='lower center')
    save(fig, '07_monthly_power', book)
    fig, axs = plt.subplots(1, 3, figsize=(11.9, 3.6))
    fig.subplots_adjust(left=.05, right=.985, bottom=.18, top=.87, wspace=.25)
    for ax, q in zip(axs, (1,2,3)):
        for key, label, color, marker in zip(('cosine','shadow_block','intercept'),
                    ('余弦效率','阴影遮挡效率','截断效率'), COLORS, ('s','^','D')):
            ax.plot(months, monthly[q][key], label=label, color=color, marker=marker, markersize=3)
        ax.set(title=NAMES[q], xlabel='月份', ylabel='面积加权平均效率', ylim=(.67,1.005),
               xticks=[1,3,6,9,12])
        ax.grid(alpha=.18)
    axs[1].legend(ncol=3, bbox_to_anchor=(.5,-.21), loc='upper center', fontsize=9)
    save(fig, '08_monthly_efficiencies', book)


def build(entries):
    configure()
    norm = Normalize(.15,.75)
    with PdfPages(HERE / 'all_figures.pdf') as book:
        for q in (1,2,3):
            save(layout_figure(q, entries[q], norm), f'0{q}_q{q}_layout', book)
        save(comparison_figure(entries, norm), '04_q2_q3_comparison', book)
        save(heterogeneity_figure(entries[3]), '05_q3_area_and_height', book)
        save(radial_figure(entries[3]), '06_q3_specifications', book)
        monthly_figures(entries, book)
        save(detail_figure(entries[3]), '09_q3_rectangle_detail', book)
    for q in (1,2,3):
        save(layout_figure(q, entries[q], norm, 1.), f'to_scale/q{q}_layout')
    save(heterogeneity_figure(entries[3], 1.), 'to_scale/q3_area_and_height')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    protected = {str(p.relative_to(ROOT)): digest(p) for p in INFO.rglob('*') if p.is_file()}
    for name in ('final/paper.pdf','final/paper.tex','final/data/q2.npz','final/data/q3.npz'):
        protected[name] = digest(ROOT / name)
    entries = prepare_data()
    if not args.data_only:
        build(entries)
    assert all(digest(ROOT / p) == h for p,h in protected.items())
    write_json(HERE / 'manifest.json', dict(input_sha256=protected, unchanged=True,
        figure_script_sha256=digest(Path(__file__)),
        reference_pose='March 21, solar time 12:00; true horizontal projection',
        compact_symbol_linear_scale=SCALE, exact_scale_alternatives='to_scale/',
        geometric_coordinates_changed=False, rays_per_mirror_time=RAYS, seed=SEED,
        outputs=OUTPUTS, summaries={q: e[4] for q,e in entries.items()}))
    print(f'Completed: {len(OUTPUTS)} figures, each in PDF / SVG / 360-dpi PNG.', flush=True)


if __name__ == '__main__':
    main()
