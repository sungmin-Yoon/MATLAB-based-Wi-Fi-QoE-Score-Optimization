import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모든 데이터 합치기
df1 = pd.read_excel(r'C:\Users\kimse\OneDrive\Desktop\hope\wifi_data.xlsx').dropna().reset_index(drop=True)
df2 = pd.read_excel(r'C:\Users\kimse\OneDrive\Desktop\hope\wifi_data2.xlsx').dropna().reset_index(drop=True)
df2.columns = df1.columns
df_all = pd.concat([df1, df2], ignore_index=True)

cols = df1.columns.tolist()
RSSI_col = df_all[cols[0]]   # RSSI
Link_col = df_all[cols[1]]   # Link
Util_col = df_all[cols[2]]   # 채널사용률
RTT_col  = df_all[cols[3]]   # RTT
Loss_col = df_all[cols[4]]   # 패킷손실

print("▶ RSSI와 각 지표 간 상관계수 (전체 데이터 기준)")
print(f"  RSSI ↔ Link:   {df_all[cols[0]].corr(df_all[cols[1]]):.3f}")
print(f"  RSSI ↔ RTT:    {df_all[cols[0]].corr(df_all[cols[3]]):.3f}")
print(f"  RSSI ↔ Util:   {df_all[cols[0]].corr(df_all[cols[2]]):.3f}")
print(f"  RSSI ↔ Loss:   {df_all[cols[0]].corr(df_all[cols[4]]):.3f}")
print(f"  RSSI ↔ Speed(DL+UL): {df_all[cols[0]].corr(df_all[cols[6]]+df_all[cols[5]]):.3f}")

print("\n▶ wifi_data1 (평상시)")
print(f"  RSSI ↔ Link:   {df1[cols[0]].corr(df1[cols[1]]):.3f}")
print(f"  RSSI ↔ RTT:    {df1[cols[0]].corr(df1[cols[3]]):.3f}")
print(f"  RSSI ↔ Util:   {df1[cols[0]].corr(df1[cols[2]]):.3f}")

print("\n▶ wifi_data2 (혼잡)")
print(f"  RSSI ↔ Link:   {df2[cols[0]].corr(df2[cols[1]]):.3f}")
print(f"  RSSI ↔ RTT:    {df2[cols[0]].corr(df2[cols[3]]):.3f}")
print(f"  RSSI ↔ Util:   {df2[cols[0]].corr(df2[cols[2]]):.3f}")

# ── 그래프 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("신호 세기(RSSI)와 각 지표의 관계 (전체 데이터 기준)", fontsize=16, fontweight='bold')

pairs = [
    (RTT_col,  "RTT (지연시간, ms)",    "낮을수록 좋음", '#e74c3c'),
    (Link_col, "Link Speed (Mbps)",     "높을수록 좋음", '#3498db'),
    (Util_col, "채널사용률 (%)",         "낮을수록 좋음", '#e67e22'),
    (Loss_col, "패킷손실률 (%)",         "낮을수록 좋음", '#9b59b6'),
]

for ax, (y_data, ylabel, note, color) in zip(axes.flat, pairs):
    corr = RSSI_col.corr(y_data)
    ax.scatter(RSSI_col, y_data, alpha=0.4, s=30, color=color)

    # 추세선
    z = np.polyfit(RSSI_col, y_data, 1)
    p = np.poly1d(z)
    x_line = np.linspace(RSSI_col.min(), RSSI_col.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("신호 세기 (RSSI, dBm) — 오른쪽일수록 강함", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"RSSI vs {ylabel}\n상관계수: {corr:.3f}  ({note})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(0.05, 0.93, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            color='green' if abs(corr) > 0.3 else 'gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(r'C:\Users\kimse\OneDrive\Desktop\hope\rssi_influence.png', dpi=300, bbox_inches='tight')
print("\n✅ 저장: rssi_influence.png")
