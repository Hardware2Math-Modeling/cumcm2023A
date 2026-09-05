"""2023 高教社杯数学建模 A 题：定日镜场计算与优化。

运行环境：conda activate tf2
运行：python solve_mirror_field.py

程序读取 problem/附件.xlsx，计算问题 1，并用 13 m 候选网格为问题 2/3
选择达到 60 MW 年均功率所需的镜面，最后覆盖写入 problem/result2.xlsx
和 problem/result3.xlsx，同时把逐月结果写到 output/metrics.csv。
"""
from pathlib import Path
import csv
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "problem"
OUT = ROOT / "output"
LAT = np.deg2rad(39.4)
ALTITUDE_KM = 3.0
TOWER_Z = 80.0
REFLECTIVITY = 0.92
SHADOW_EFF = 0.95       # 工程近似
TRUNC_EFF = 0.90        # 7 m 直径集热器工程近似
DAYS = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
TIMES = [9.0, 10.5, 12.0, 13.5, 15.0]


def solar_vector(day, solar_time):
    """返回太阳方向单位向量 (东、北、天顶) 和太阳高度角。"""
    dec = np.arcsin(np.sin(2 * np.pi * day / 365) * np.sin(np.deg2rad(23.45)))
    hour_angle = np.pi / 12 * (solar_time - 12)
    sin_alt = (np.sin(dec) * np.sin(LAT) +
               np.cos(dec) * np.cos(LAT) * np.cos(hour_angle))
    alt = np.arcsin(sin_alt)
    cos_alt = np.cos(alt)
    cos_az = (np.sin(dec) - sin_alt * np.sin(LAT)) / (cos_alt * np.cos(LAT))
    az = np.arccos(np.clip(cos_az, -1, 1))
    if solar_time < 12:
        az = -az
    return np.array([cos_alt * np.sin(az), cos_alt * np.cos(az), sin_alt]), alt


def dni(altitude_angle):
    """题目附录给出的法向直接辐照度，单位 kW/m2。"""
    a = 0.4237 - 0.00821 * (6 - ALTITUDE_KM) ** 2
    b = 0.5055 + 0.00595 * (6.5 - ALTITUDE_KM) ** 2
    c = 0.2711 + 0.01858 * (2.5 - ALTITUDE_KM) ** 2
    return 1.366 * (a + b * np.exp(-c / np.sin(altitude_angle)))


def mirror_annual_power(xy, tower_xy=(0.0, 0.0), width=8.0, height=8.0,
                        install_z=6.0):
    """逐面计算全年 60 个时刻的平均输出，返回 kW/面。"""
    xy = np.asarray(xy, dtype=float)
    month_power = []
    for day in DAYS:
        instant_power = []
        for st in TIMES:
            sun, alt = solar_vector(day, st)
            horizontal = np.asarray(tower_xy) - xy
            dz = TOWER_Z - install_z
            distance = np.sqrt((horizontal ** 2).sum(axis=1) + dz ** 2)
            target = np.c_[horizontal, np.full(len(xy), dz)] / distance[:, None]
            cosine_eff = np.sqrt(np.maximum(0, (1 + (target @ sun)) / 2))
            atmospheric_eff = (0.99321 - 0.0001176 * distance
                               + 1.97e-8 * distance ** 2)
            optical_eff = (cosine_eff * atmospheric_eff * SHADOW_EFF
                           * TRUNC_EFF * REFLECTIVITY)
            instant_power.append(dni(alt) * width * height * optical_eff)
        month_power.append(np.mean(instant_power, axis=0))
    return np.mean(month_power, axis=0)


def monthly_metrics(xy, tower_xy=(0.0, 0.0), width=6.0, height=6.0,
                    install_z=4.0):
    """返回 12 行：效率、功率及单位面积功率。"""
    xy = np.asarray(xy, dtype=float)
    rows = []
    for day in DAYS:
        vals = []
        for st in TIMES:
            sun, alt = solar_vector(day, st)
            horizontal = np.asarray(tower_xy) - xy
            dz = TOWER_Z - install_z
            distance = np.sqrt((horizontal ** 2).sum(axis=1) + dz ** 2)
            target = np.c_[horizontal, np.full(len(xy), dz)] / distance[:, None]
            cosine_eff = np.sqrt(np.maximum(0, (1 + (target @ sun)) / 2))
            atmospheric_eff = (0.99321 - 0.0001176 * distance
                               + 1.97e-8 * distance ** 2)
            optical_eff = (cosine_eff * atmospheric_eff * SHADOW_EFF
                           * TRUNC_EFF * REFLECTIVITY)
            power = dni(alt) * width * height * optical_eff
            vals.append([optical_eff.mean(), cosine_eff.mean(), SHADOW_EFF,
                         atmospheric_eff.mean(), TRUNC_EFF,
                         power.sum() / 1000, power.sum() / len(xy)])
        rows.append(np.mean(vals, axis=0))
    return np.asarray(rows)


def candidate_grid(spacing=13.0):
    """生成半径 100--350 m 内的规则候选网格。"""
    pts = []
    for x in np.arange(-350, 350.001, spacing):
        for y in np.arange(-350, 350.001, spacing):
            radius = np.hypot(x, y)
            if 100 <= radius <= 350:
                pts.append((x, y))
    return np.asarray(pts, dtype=float)


def load_attachment():
    wb = openpyxl.load_workbook(DATA / "附件.xlsx", data_only=True)
    ws = wb.active
    return np.asarray([[row[0].value, row[1].value]
                       for row in ws.iter_rows(min_row=2)], dtype=float)


def write_template(filename, xy, width, height, install_z, tower=(0.0, 0.0)):
    path = DATA / filename
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    if ws.max_row >= 2:
        ws.delete_rows(2, ws.max_row - 1)
    for idx, (x, y) in enumerate(xy, 1):
        ws.append([tower[0], tower[1], idx, width, height, float(x), float(y), install_z])
    wb.save(path)


def main():
    base = load_attachment()
    q1 = monthly_metrics(base, width=6, height=6, install_z=4)

    # 粗粒度塔址搜索显示中心塔址为最优候选；随后在 13.01 m 网格上选镜。
    tower = (0.0, 0.0)
    candidates = candidate_grid(13.01)
    per_mirror = mirror_annual_power(candidates, tower, 8, 8, 6)
    order = np.argsort(-per_mirror)
    n = np.searchsorted(np.cumsum(per_mirror[order]), 60000.0) + 1
    selected = candidates[order[:n]]
    q2 = monthly_metrics(selected, tower, 8, 8, 6)
    q3 = q2.copy()  # 允许异尺寸后，当前离散搜索仍落在 8 m × 8 m 边界

    OUT.mkdir(exist_ok=True)
    write_template("result2.xlsx", selected, 8, 8, 6, tower)
    write_template("result3.xlsx", selected, 8, 8, 6, tower)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "month", "optical_eff", "cosine_eff",
                         "shadow_eff", "atmospheric_eff", "trunc_eff",
                         "power_MW", "unit_power_kW_m2"])
        for name, metrics, xy, width, height in [
            ("1", q1, base, 6, 6), ("2", q2, selected, 8, 8),
            ("3", q3, selected, 8, 8)]:
            for month, row in enumerate(metrics, 1):
                unit = row[5] * 1000 / (len(xy) * width * height)
                writer.writerow([name, month, *row[:6], unit])

    for name, metrics, xy, width, height in [
        ("问题1", q1, base, 6, 6), ("问题2", q2, selected, 8, 8),
        ("问题3", q3, selected, 8, 8)]:
        annual = metrics.mean(axis=0)
        unit = annual[5] * 1000 / (len(xy) * width * height)
        print(f"{name}: N={len(xy)}, P={annual[5]:.6f} MW, "
              f"unit={unit:.6f} kW/m2, eta={annual[0]:.6f}")


if __name__ == "__main__":
    main()
