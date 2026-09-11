import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HOPE_DIR = r'C:\Users\kimse\OneDrive\Desktop\hope'
COMP_DIR = r'C:\Users\kimse\OneDrive\Desktop\HOPE상황비교'

# ══════════════════════════════════════════════════════════════
# 하이브리드 임계값 설정
# ══════════════════════════════════════════════════════════════
TH_RTT  = 22   # ms  이상이면 혼잡 모드
TH_UTIL = 56   # %   이상이면 혼잡 모드
W_RTT   = 0.60 # 혼잡 모드 RTT 가중치
W_UTIL  = 0.40 # 혼잡 모드 채널사용률 가중치

# ══════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════
df1_all = pd.read_excel(f'{HOPE_DIR}/wifi_data.xlsx').dropna().reset_index(drop=True)
df2_all = pd.read_excel(f'{HOPE_DIR}/wifi_data2.xlsx').dropna().reset_index(drop=True)
df2_all.columns = df1_all.columns

# 각 60개씩만 사용
df1 = df1_all.iloc[:60].copy()
df2 = df2_all.iloc[:60].copy()
df1r = df1.copy(); df1r.columns = range(df1.shape[1])
df2r = df2.copy(); df2r.columns = range(df2.shape[1])
df_comb = pd.concat([df1r, df2r], ignore_index=True)  # 120개

df_cong = pd.read_excel(f'{COMP_DIR}/오열_트래픽과부화상황.xlsx').dropna().reset_index(drop=True).iloc[:45]
df_norm = pd.read_excel(f'{COMP_DIR}/오열에서_오픈열람실.xlsx').dropna().reset_index(drop=True).iloc[:45]
df_jay  = pd.read_excel(f'{COMP_DIR}/오열에서_자유열람실.xlsx').dropna().reset_index(drop=True).iloc[:45]

# ══════════════════════════════════════════════════════════════
# 전역 정규화 범위 (wifi_data1 전체 기준 고정)
# ══════════════════════════════════════════════════════════════
# 전역 정규화 범위: 원본 전체 데이터 기준 고정 (60개 샘플 bias 방지)
G = {
    'rssi': (df1_all.iloc[:,0].min(), df1_all.iloc[:,0].max()),
    'rtt':  (df1_all.iloc[:,3].min(), df1_all.iloc[:,3].max()),
    'util': (df1_all.iloc[:,2].min(), df1_all.iloc[:,2].max()),
}

def norm(s, lo, hi, invert=False):
    if hi == lo: return pd.Series(0.5, index=s.index)
    v = ((s - lo) / (hi - lo)).clip(0, 1)
    return 1 - v if invert else v

def featurize(df):
    return {
        'P_RSSI': norm(df.iloc[:,0], *G['rssi']),
        'P_RTT':  norm(df.iloc[:,3], *G['rtt'],  invert=True),
        'P_UTIL': norm(df.iloc[:,2], *G['util'],  invert=True),
        'RTT_raw':  df.iloc[:,3].values,
        'UTIL_raw': df.iloc[:,2].values,
        'RSSI_raw': df.iloc[:,0].values,
        'y_true':   df.iloc[:,6] + df.iloc[:,5],
    }

# ══════════════════════════════════════════════════════════════
# 하이브리드 점수 계산
# ══════════════════════════════════════════════════════════════
def hybrid_score(feats):
    """
    혼잡 조건: RTT >= TH_RTT OR Util >= TH_UTIL
    - 혼잡: RTT*0.6 + Util*0.4 (채널 실제 상태 기반)
    - 정상: RSSI 100% (신호 세기 기반)
    """
    rtt_arr  = feats['RTT_raw']
    util_arr = feats['UTIL_raw']
    is_congested = (rtt_arr >= TH_RTT) | (util_arr >= TH_UTIL)

    scores = np.where(
        is_congested,
        (W_RTT  * feats['P_RTT']  + W_UTIL * feats['P_UTIL'])  * 100,
        feats['P_RSSI'] * 100
    )
    return pd.Series(scores.clip(0, 100), index=feats['P_RSSI'].index), is_congested

# ══════════════════════════════════════════════════════════════
# ① 산점도 3장 (일반 / 혼잡 / 통합)
# ══════════════════════════════════════════════════════════════
print("▶ 산점도 생성 중...")
scenarios = [
    (featurize(df1r),    "Case 1: 일반 환경",  f"wifi_data1 ({len(df1r)}개)"),
    (featurize(df2r),    "Case 2: 혼잡 환경",  f"wifi_data2 ({len(df2r)}개)"),
    (featurize(df_comb), "Case 3: 통합 환경",  f"60+60 = {len(df_comb)}개"),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 19))
fig.suptitle(f"HOPE 하이브리드 모드\n(혼잡 감지: RTT≥{TH_RTT}ms OR 채널사용률≥{TH_UTIL}%)",
             fontsize=16, fontweight='bold', y=1.01)

for row, (feats, case_title, data_label) in enumerate(scenarios):
    y = feats['y_true']
    base  = feats['P_RSSI'] * 100
    hope, is_cong = hybrid_score(feats)

    cb = np.corrcoef(base, y)[0,1]
    ch = np.corrcoef(hope, y)[0,1]
    diff = ch - cb
    n_cong = is_cong.sum()
    n_norm = (~is_cong).sum()

    # 왼쪽: 기존 RSSI
    ax = axes[row, 0]
    sns.regplot(x=y, y=base, ax=ax, scatter_kws={'alpha':0.55,'s':50}, color='#e74c3c')
    ax.set_title(f"{case_title} — [{data_label}]\n기존 방식 (RSSI 100%)  r={cb:.3f}", fontsize=12, fontweight='bold')
    ax.set_xlabel("실제 속도 (Down+Up Mbps)", fontsize=10)
    ax.set_ylabel("AP 점수", fontsize=10)
    ax.set_xlim(-5, y.max()*1.1); ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 오른쪽: 하이브리드
    ax = axes[row, 1]
    # 정상 모드 점 (파랑), 혼잡 모드 점 (주황)
    ax.scatter(y[~is_cong], hope[~is_cong], alpha=0.6, s=50,
               color='#3498db', label=f'정상 모드 (RSSI): {n_norm}개')
    ax.scatter(y[is_cong], hope[is_cong], alpha=0.6, s=60, marker='D',
               color='#e67e22', label=f'혼잡 모드 (RTT+Util): {n_cong}개')

    # 추세선 전체
    z = np.polyfit(y, hope, 1)
    x_line = np.linspace(y.min(), y.max(), 200)
    ax.plot(x_line, np.poly1d(z)(x_line), color='#2ecc71', linewidth=2.5, linestyle='--')

    ax.set_title(f"{case_title} — [{data_label}]\nHOPE 하이브리드  r={ch:.3f}  ({'+' if diff>=0 else ''}{diff*100:.1f}%p)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("실제 속도 (Down+Up Mbps)", fontsize=10)
    ax.set_ylabel("AP 점수 (하이브리드)", fontsize=10)
    ax.set_xlim(-5, y.max()*1.1); ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    print(f"  {case_title}: 기존={cb:.3f} → 하이브리드={ch:.3f}  ({'+' if diff>=0 else ''}{diff*100:.1f}%p)  혼잡감지={n_cong}/{len(y)}")

plt.tight_layout()
plt.savefig(f'{HOPE_DIR}/hybrid_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 저장: hybrid_scatter.png")

# ══════════════════════════════════════════════════════════════
# ② HOPE상황비교 Cycle 4분할 그래프 (하이브리드 적용)
# ══════════════════════════════════════════════════════════════
print("\n▶ Cycle 4분할 그래프 생성 중...")
fc = featurize(df_cong)
fn = featurize(df_norm)
fj = featurize(df_jay)

# 하이브리드: 오열(현재 연결 AP) 기준으로 혼잡 여부 판단
# 경우1: 오열이 터짐 → 오열의 RTT/Util로 판단
# 경우2: 오열이 정상 → 마찬가지로 오열 기준 판단
def cycle_hybrid(df_current, df_other, feats_cur, feats_oth):
    """
    현재 연결 AP(current)가 혼잡하면 → 상대방(other)과 RTT+Util 점수로 비교
    정상이면 → RSSI 점수로 비교 (그대로 유지)
    """
    rtt_c  = df_current.iloc[:,3].values
    util_c = df_current.iloc[:,2].values
    is_cong = (rtt_c >= TH_RTT) | (util_c >= TH_UTIL)

    # 현재 AP 점수
    score_cur = np.where(
        is_cong,
        (W_RTT*feats_cur['P_RTT'] + W_UTIL*feats_cur['P_UTIL']) * 100,
        feats_cur['P_RSSI'] * 100
    )
    # 상대방 AP 점수 (동일 기준 적용)
    score_oth = np.where(
        is_cong,
        (W_RTT*feats_oth['P_RTT'] + W_UTIL*feats_oth['P_UTIL']) * 100,
        feats_oth['P_RSSI'] * 100
    )
    return (pd.Series(score_cur.clip(0,100)),
            pd.Series(score_oth.clip(0,100)),
            is_cong)

sc1, sj1, cong1 = cycle_hybrid(df_cong, df_jay,  fc, fj)  # 경우1
sc2, sj2, cong2 = cycle_hybrid(df_norm, df_jay,  fn, fj)  # 경우2

bc = fc['P_RSSI']*100; bj = fj['P_RSSI']*100
bn = fn['P_RSSI']*100

print(f"  경우1 혼잡감지: {cong1.sum()}/45 사이클")
print(f"  경우2 혼잡감지: {cong2.sum()}/45 사이클")
print(f"  경우1: 오열={sc1.mean():.1f} vs 자열={sj1.mean():.1f}  → {'자열 우위 ✅' if sj1.mean()>sc1.mean() else '오열 우위'}")
print(f"  경우2: 오열={sc2.mean():.1f} vs 자열={sj2.mean():.1f}  → {'오열 우위 ✅' if sc2.mean()>sj2.mean() else '자열 우위'}")

N = 45; cycles = np.arange(1, N+1)
RED, BLUE = '#e74c3c', '#3498db'

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"HOPE 하이브리드 — 상황별 AP 선택\n(혼잡 감지 조건: RTT ≥ {TH_RTT}ms OR 채널사용률 ≥ {TH_UTIL}%)",
             fontsize=16, fontweight='bold', y=1.01)

def draw_cycle(ax, y_r, y_b, title, ylabel, is_cong=None, ylim=(0,105)):
    yr, yb = np.array(y_r), np.array(y_b)
    ax.plot(cycles, yr, color=RED,  linewidth=2.5, label='오픈열람실 (오열)')
    ax.plot(cycles, yb, color=BLUE, linewidth=2.5, label='자유열람실 (자열)')
    ax.fill_between(cycles, yr, yb, where=(yb>yr), alpha=0.15, color=BLUE)
    ax.fill_between(cycles, yr, yb, where=(yr>=yb), alpha=0.15, color=RED)
    # 혼잡 모드 구간 표시
    if is_cong is not None:
        for i, cong in enumerate(is_cong):
            if cong:
                ax.axvspan(i+0.5, i+1.5, alpha=0.12, color='#f39c12', zorder=0)
        orange_patch = mpatches.Patch(color='#f39c12', alpha=0.4, label='혼잡 모드 전환 구간')
        ax.legend(handles=[
            plt.Line2D([0],[0],color=RED,lw=2,label='오픈열람실 (오열)'),
            plt.Line2D([0],[0],color=BLUE,lw=2,label='자유열람실 (자열)'),
            orange_patch], fontsize=9, loc='upper right')
    else:
        ax.legend(fontsize=10, loc='upper right')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Cycle (시행 횟수)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(1, N); ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.5)

draw_cycle(axs[0,0], sc1, sj1,
    f"경우1 — HOPE 하이브리드 점수 [80~100]\n혼잡 감지 {cong1.sum()}/45 → 자열 추천!",
    "AP 점수", is_cong=cong1, ylim=(80, 100))

draw_cycle(axs[1,0], bc, bj,
    f"경우1 — 기존 방식 (RSSI만)\n오열 신호 더 강함 → 기존은 계속 오열 선택",
    "RSSI 기반 점수", ylim=(0, 105))

draw_cycle(axs[0,1], sc2, sj2,
    f"경우2 — HOPE 하이브리드 점수 [80~100]\n혼잡 감지 {cong2.sum()}/45 → 대부분 오열 유지",
    "AP 점수", is_cong=cong2, ylim=(80, 100))

draw_cycle(axs[1,1], bn, bj,
    f"경우2 — 기존 방식 (RSSI만)\n오열 신호 더 강함 → 기존도 오열 선택",
    "RSSI 기반 점수", ylim=(0, 105))

plt.tight_layout()
plt.savefig(f'{COMP_DIR}/hybrid_cycle.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 저장: hybrid_cycle.png")

print("\n✅ 모두 완료!")
print(f"  임계값: RTT≥{TH_RTT}ms OR 채널사용률≥{TH_UTIL}%")
print(f"  혼잡 모드 가중치: RTT {W_RTT*100:.0f}% : 채널사용률 {W_UTIL*100:.0f}%")
