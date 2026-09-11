import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HOPE_DIR = r'C:\Users\kimse\OneDrive\Desktop\hope'

df1 = pd.read_excel(f'{HOPE_DIR}/wifi_data.xlsx').dropna().reset_index(drop=True)
df2 = pd.read_excel(f'{HOPE_DIR}/wifi_data2.xlsx').dropna().reset_index(drop=True)
df2.columns = df1.columns

rtt1  = df1.iloc[:,3];  rtt2  = df2.iloc[:,3]
util1 = df1.iloc[:,2];  util2 = df2.iloc[:,2]

# ── 통계 출력 ──
print("=" * 55)
print("  RTT (지연시간, ms)")
print("=" * 55)
for label, s in [("일반(data1)", rtt1), ("혼잡(data2)", rtt2)]:
    p25, p50, p75 = np.percentile(s, [25,50,75])
    print(f"  {label}: 평균={s.mean():.1f}, 중앙={p50:.1f}, "
          f"75%ile={p75:.1f}, 90%ile={np.percentile(s,90):.1f}, 최대={s.max():.1f}")

print()
print("=" * 55)
print("  채널사용률 (%)")
print("=" * 55)
for label, s in [("일반(data1)", util1), ("혼잡(data2)", util2)]:
    p25, p50, p75 = np.percentile(s, [25,50,75])
    print(f"  {label}: 평균={s.mean():.1f}, 중앙={p50:.1f}, "
          f"75%ile={p75:.1f}, 90%ile={np.percentile(s,90):.1f}, 최대={s.max():.1f}")

# ── 임계값별 분류 정확도 계산 ──
print()
print("=" * 55)
print("  RTT 임계값별 혼잡 감지 정확도")
print("=" * 55)
# 정상=0, 혼잡=1 레이블
all_rtt  = pd.concat([rtt1, rtt2], ignore_index=True)
labels   = np.array([0]*len(rtt1) + [1]*len(rtt2))
best_rtt_th, best_rtt_acc = 0, 0
for th in range(5, 70, 1):
    pred = (all_rtt >= th).astype(int)
    acc = (pred.values == labels).mean()
    if acc > best_rtt_acc:
        best_rtt_acc, best_rtt_th = acc, th

for th in [10, 15, 20, 25, 30, 35, best_rtt_th]:
    pred = (all_rtt >= th).astype(int)
    acc  = (pred.values == labels).mean()
    tp   = ((pred.values==1) & (labels==1)).sum()
    fp   = ((pred.values==1) & (labels==0)).sum()
    fn   = ((pred.values==0) & (labels==1)).sum()
    print(f"  RTT >= {th:3d}ms → 정확도={acc*100:.1f}%  감지={tp}/{len(rtt2)}  오감지={fp}/{len(rtt1)}")

print()
print("=" * 55)
print("  채널사용률 임계값별 혼잡 감지 정확도")
print("=" * 55)
all_util = pd.concat([util1, util2], ignore_index=True)
best_u_th, best_u_acc = 0, 0
for th in range(20, 90, 1):
    pred = (all_util >= th).astype(int)
    acc = (pred.values == labels).mean()
    if acc > best_u_acc:
        best_u_acc, best_u_th = acc, th

for th in [30, 40, 50, 55, 60, best_u_th]:
    pred = (all_util >= th).astype(int)
    acc  = (pred.values == labels).mean()
    tp   = ((pred.values==1) & (labels==1)).sum()
    fp   = ((pred.values==1) & (labels==0)).sum()
    fn   = ((pred.values==0) & (labels==1)).sum()
    print(f"  Util >= {th:3d}%  → 정확도={acc*100:.1f}%  감지={tp}/{len(util2)}  오감지={fp}/{len(util1)}")

print(f"\n  최적: RTT >= {best_rtt_th}ms (정확도 {best_rtt_acc*100:.1f}%)")
print(f"  최적: Util >= {best_u_th}% (정확도 {best_u_acc*100:.1f}%)")

# ── 시각화 ──
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("일반 vs 혼잡 환경 지표 분포 — 임계값 탐색", fontsize=16, fontweight='bold')

# RTT 분포
ax = axes[0]
ax.hist(rtt1, bins=30, alpha=0.6, color='#3498db', label=f'일반(data1, n={len(rtt1)})', density=True)
ax.hist(rtt2, bins=30, alpha=0.6, color='#e74c3c', label=f'혼잡(data2, n={len(rtt2)})', density=True)
ax.axvline(best_rtt_th, color='black', linestyle='--', linewidth=2,
           label=f'최적 임계값: {best_rtt_th}ms (정확도 {best_rtt_acc*100:.1f}%)')
ax.axvspan(best_rtt_th, rtt2.max()*1.1, alpha=0.07, color='red', label='혼잡 모드 구간')
ax.set_xlabel("RTT (ms)", fontsize=12)
ax.set_ylabel("밀도", fontsize=12)
ax.set_title("RTT 분포 — 일반 vs 혼잡", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# 채널사용률 분포
ax = axes[1]
ax.hist(util1, bins=25, alpha=0.6, color='#3498db', label=f'일반(data1, n={len(util1)})', density=True)
ax.hist(util2, bins=25, alpha=0.6, color='#e74c3c', label=f'혼잡(data2, n={len(util2)})', density=True)
ax.axvline(best_u_th, color='black', linestyle='--', linewidth=2,
           label=f'최적 임계값: {best_u_th}% (정확도 {best_u_acc*100:.1f}%)')
ax.axvspan(best_u_th, 100, alpha=0.07, color='red', label='혼잡 모드 구간')
ax.set_xlabel("채널사용률 (%)", fontsize=12)
ax.set_ylabel("밀도", fontsize=12)
ax.set_title("채널사용률 분포 — 일반 vs 혼잡", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{HOPE_DIR}/threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n저장: threshold_analysis.png")
