import time, warnings
from datetime import datetime, timezone, timedelta
import yfinance as yf, pandas as pd, numpy as np

warnings.filterwarnings("ignore")

# ==========================================
# 核心设置：全新起跑日期（从这一天开始 0 仓位、$10万现金起步）
# ==========================================
LIVE_START = "2026-11-19"
INIT_CAP = 100000.0

MY_50 = [
    "NOW","PANW","CRWD","ANET","VEEV","TTD","FICO","PLTR","DDOG","AMD",
    "SNPS","ISRG","BSX","IDXX","VRTX","AMZN","TSLA","NFLX","UBER","MELI",
    "CMG","BRK-B","AXON","MKL","CPRT",
    "NEE","CVX","XOM","SO","HD","LOW","COST","PG","V","MSFT",
    "MA","JPM","SPGI","AVGO","QCOM","TXN","KLAC","ABBV","UNH","SYK",
    "LLY","LMT","UNP","WM","ADP"
]

N_HIGH, MAX_SLOTS, SL_ATR, TRAIL_ATR, TIME_STOP = 20, 12, 3.5, 4.5, 45
TARGET_VOL, SLIP, D_CASH = 0.14, 0.00075, 0.04 / 252.0

print("🌐 拉取 50 股 + SPY + QQQ + GLD 行情（用于计算指标）...")
data_dict, all_t = {}, MY_50 + ["SPY", "QQQ", "GLD"]
for i in range(0, len(all_t), 10):
    batch = all_t[i:i+10]
    df_b = yf.download(batch, start="2025-01-01", interval="1d", group_by="ticker", progress=False, threads=False, timeout=20)
    for t in batch:
        try:
            sub = df_b[t].dropna() if len(batch) > 1 else df_b.dropna()
            if isinstance(sub.columns, pd.MultiIndex): sub.columns = sub.columns.get_level_values(0)
            if len(sub) > 100: data_dict[t] = sub[['Open','High','Low','Close','Volume']].astype(np.float64)
        except Exception: pass
    time.sleep(0.3)

dates = data_dict["SPY"].index
T, S = len(dates), len(MY_50)
spy_c = data_dict["SPY"]["Close"].reindex(dates).ffill().values
qqq_c = data_dict["QQQ"]["Close"].reindex(dates).ffill().values
gld_c = data_dict["GLD"]["Close"].reindex(dates).ffill().values
gld_r = np.zeros(T); gld_r[1:] = gld_c[1:] / gld_c[:-1] - 1.0
gld_bull = gld_c > pd.Series(gld_c).rolling(60).mean().values

O_arr, C_arr = np.full((S, T), np.nan), np.full((S, T), np.nan)
ATR_arr, pos52_arr = np.zeros((S, T)), np.full((S, T), 0.5)
valid_arr, entry_arr = np.zeros((S, T), bool), np.zeros((S, T), bool)

for i, t in enumerate(MY_50):
    if t not in data_dict: continue
    df = data_dict[t].reindex(dates)
    fv = df['Close'].first_valid_index()
    if fv is None: continue
    sub = df.loc[fv:].ffill()
    idx0 = dates.get_loc(fv)
    o, h, l, c, v = sub['Open'].values, sub['High'].values, sub['Low'].values, sub['Close'].values, sub['Volume'].values
    O_arr[i, idx0:], C_arr[i, idx0:] = o, c
    sc, sh, sl, sv = pd.Series(c), pd.Series(h), pd.Series(l), pd.Series(v)
    tr = pd.concat([sh - sl, (sh - sc.shift(1)).abs(), (sl - sc.shift(1)).abs()], axis=1).max(axis=1)
    ATR_arr[i, idx0:] = tr.rolling(20).mean().fillna(0).values
    ma60 = sc.rolling(60).mean()
    below = (sc < ma60).astype(float)
    entry_arr[i, idx0:] = ((below.rolling(40).sum().shift(1) >= 10) & (sc > ma60) & (below.rolling(5).max().shift(1) == 1.0) & (sc > sc.rolling(N_HIGH).max().shift(1)) & (sv >= sv.rolling(20).mean())).fillna(False).values
    pos52_arr[i, idx0:] = ((sc - sl.rolling(252).min()) / (sh.rolling(252).max() - sl.rolling(252).min() + 1e-6)).fillna(0.5).values
    if idx0 + 60 < T: valid_arr[i, idx0 + 60:] = True

# 1. 从 LIVE_START 起全新 0 仓位记账
start_idx = int(np.searchsorted(dates, pd.Timestamp(LIVE_START)))
cash, holdings, trade_logs = 1.0, {}, []
eq_v0, r_v0 = np.ones(T), np.zeros(T)

for t in range(start_idx, T):
    v_now = valid_arr[:, t] & valid_arr[:, t-1]
    cash *= (1.0 + D_CASH)
    to_sell = []
    for i, p in list(holdings.items()):
        pc, patr = float(C_arr[i, t-1]), float(ATR_arr[i, t-1])
        p['peak'] = max(p['peak'], pc); p['days'] += 1
        rsn = None
        if not v_now[i]: rsn = "停牌/缺数据"
        elif pc < p['init_sl']: rsn = f"ATR止损({SL_ATR}x)"
        elif pc < p['peak'] - TRAIL_ATR * patr: rsn = f"吊灯止盈({TRAIL_ATR}x)"
        elif p['days'] >= TIME_STOP and pc <= p['ep']: rsn = "45天时间止损"
        if rsn:
            sp = float(O_arr[i, t])
            cash += p['sh'] * sp * (1.0 - SLIP)
            trade_logs.append({"t": MY_50[i].replace("-","."), "bp": round(p['ep'],2), "sd": dates[t].strftime("%m-%d"), "sp": round(sp,2), "pnl": round((sp*(1-2*SLIP)/p['ep']-1)*100,1), "rsn": rsn})
            to_sell.append(i)
    for i in to_sell: del holdings[i]

    eq_open = cash + sum(p['sh'] * float(O_arr[i, t]) for i, p in holdings.items())
    free = MAX_SLOTS - len(holdings)
    if free > 0 and cash > eq_open * 0.03:
        cands = sorted([i for i in range(S) if (i not in holdings) and v_now[i] and entry_arr[i, t-1]], key=lambda idx: pos52_arr[idx, t-1])
        for i in cands[:free]:
            if cash < eq_open * 0.03: break
            alloc, ep = min(cash, eq_open / MAX_SLOTS), float(O_arr[i, t])
            holdings[i] = {"sh": alloc*(1-SLIP)/ep, "ep": ep, "bd": dates[t], "init_sl": ep - SL_ATR*float(ATR_arr[i, t-1]), "peak": ep, "days": 0}
            cash -= alloc

    eq_now = cash + sum(p['sh'] * float(C_arr[i, t]) for i, p in holdings.items())
    eq_v0[t], r_v0[t] = eq_now, eq_now / eq_v0[t-1] - 1.0

# 2. 叠加 Sub-B：波动率控制 + 黄金/短债避险（实盘未满20天时默认按目标波动率14%起步）
live_days = max(0, T - start_idx)
if live_days > 0:
    vol20_s = pd.Series(r_v0).rolling(20).std() * np.sqrt(252.0)
    w_stock = np.clip(TARGET_VOL / np.clip(vol20_s.shift(1).fillna(TARGET_VOL).values, 0.06, None), 0.30, 1.0)
    w_stock[:min(T, start_idx + 20)] = 1.0
    safe_r = np.array([gld_r[t] if (t > 0 and gld_bull[t-1]) else D_CASH for t in range(T)])
    r_subB = w_stock * r_v0 + (1.0 - w_stock) * safe_r
    eq_subB = INIT_CAP * np.cumprod(1.0 + r_subB[start_idx:])
    tot_eq, tot_ret = float(eq_subB[-1]), (float(eq_subB[-1]) / INIT_CAP - 1) * 100
    max_dd = float((eq_subB / np.maximum.accumulate(eq_subB) - 1).min() * 100)
    base_i = max(0, start_idx - 1)
    qqq_ret = (qqq_c[-1] / qqq_c[base_i] - 1) * 100
    spy_ret = (spy_c[-1] / spy_c[base_i] - 1) * 100
    cur_vol = float(vol20_s.iloc[-1]) if (live_days >= 20 and not np.isnan(vol20_s.iloc[-1])) else TARGET_VOL
else:
    tot_eq, tot_ret, max_dd, qqq_ret, spy_ret, cur_vol = INIT_CAP, 0.0, 0.0, 0.0, 0.0, TARGET_VOL

tmr_scale = float(np.clip(TARGET_VOL / max(cur_vol, 0.06), 0.30, 1.0))
tmr_safe = 1.0 - tmr_scale
safe_txt = "黄金 GLD" if gld_bull[-1] else "4% 现金短债"

# 3. 扫描下一交易日开盘信号
n_sells, n_buys = [], []
for i, p in holdings.items():
    cc, catr = float(C_arr[i, -1]), float(ATR_arr[i, -1])
    pk = max(p['peak'], cc)
    if cc < p['init_sl']: n_sells.append((MY_50[i].replace("-","."), cc, "触发ATR止损"))
    elif cc < pk - TRAIL_ATR * catr: n_sells.append((MY_50[i].replace("-","."), cc, "触发吊灯止盈"))
    elif (p['days'] + 1) >= TIME_STOP and cc <= p['ep']: n_sells.append((MY_50[i].replace("-","."), cc, "触发45天止损"))

free_tmr = MAX_SLOTS - len(holdings) + len(n_sells)
if free_tmr > 0:
    cands = sorted([i for i in range(S) if (i not in holdings) and valid_arr[i, -1] and entry_arr[i, -1]], key=lambda idx: pos52_arr[idx, -1])
    for i in cands[:free_tmr]:
        n_buys.append((MY_50[i].replace("-","."), round(float(C_arr[i, -1]), 2), round(float(pos52_arr[i, -1])*100, 1)))

# 4. 生成手机端暗黑 PWA 网页 (index.html)
bj_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%m-%d %H:%M")
mkt_date = dates[-1].strftime("%Y-%m-%d")

sig_html = f"<div class='sig macro'>⚖️ <b>明日总仓位上限</b>：股票 <b>{tmr_scale*100:.0f}%</b> + 避险 <b>{tmr_safe*100:.0f}% ({safe_txt})</b><br><span class='sub'>实盘已运行 {live_days} 个交易日 · 波动率: {cur_vol*100:.1f}%</span></div>"
for t, cc, rsn in n_sells: sig_html += f"<div class='sig sell'>🔴 <b>明日开盘平仓 {t}</b> (${cc:.2f}) · {rsn}</div>"
for t, cc, p52 in n_buys: sig_html += f"<div class='sig buy'>🟢 <b>明日开盘开仓 {t}</b> (${cc:.2f}) · 52周分位 {p52}%</div>"
if not n_sells and not n_buys: sig_html += "<div class='sig none'>☕ 明日开盘暂无新开/平仓信号，耐心等待买点。</div>"

hold_rows = []
for i, p in holdings.items():
    cc = float(C_arr[i, -1])
    ds = max(p['init_sl'], max(p['peak'], cc) - TRAIL_ATR * float(ATR_arr[i, -1]))
    pnl, dist = (cc/p['ep']-1)*100, (cc/ds-1)*100
    rw = (p['sh']*cc / eq_v0[-1]) * tmr_scale * 100
    hold_rows.append((MY_50[i].replace("-","."), p['bd'].strftime("%m-%d"), p['ep'], cc, pnl, ds, dist, rw))
hold_rows.sort(key=lambda x: x[4], reverse=True)

h_html = "".join([f"<tr><td><b>{t}</b> <span class='tag'>{rw:.1f}%</span><br><span class='sub'>{bd}</span></td><td>${ep:.2f}<br><span class='sub'>现 ${cc:.2f}</span></td><td class='{'pos' if pnl>=0 else 'neg'}'><b>{pnl:+.1f}%</b></td><td>${ds:.2f}<br><span class='sub'>{'⚠️' if dist<2 else ''}距 {dist:+.1f}%</span></td></tr>" for t,bd,ep,cc,pnl,ds,dist,rw in hold_rows])
l_html = "".join([f"<tr><td><b>{x['t']}</b><br><span class='sub'>{x['sd']}</span></td><td>${x['bp']}→${x['sp']}</td><td class='{'pos' if x['pnl']>=0 else 'neg'}'><b>{x['pnl']:+.1f}%</b></td><td class='sub'>{x['rsn']}</td></tr>" for x in reversed(trade_logs[-12:])])

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="theme-color" content="#0e1117">
<title>Sub-B 量化模拟盘</title><style>
body{{margin:0;padding:14px;background:#0e1117;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
h1{{margin:0;font-size:18px;color:#00e5ff}}.sub{{font-size:11px;color:#848e9c}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}}
.card{{background:#161b26;border:1px solid #262e3d;border-radius:10px;padding:11px}}
.val{{font-size:19px;font-weight:700;margin-top:4px}}.pos{{color:#00e676}}.neg{{color:#ff5252}}
.sec{{font-size:14px;font-weight:700;margin:15px 0 8px;color:#cbd5e1}}
.sig{{padding:10px 12px;border-radius:8px;margin-bottom:8px;font-size:13px;line-height:1.4}}
.sig.macro{{background:rgba(0,229,255,0.1);border-left:3px solid #00e5ff}}
.sig.buy{{background:rgba(0,230,118,0.12);border-left:3px solid #00e676}}
.sig.sell{{background:rgba(255,82,82,0.12);border-left:3px solid #ff5252}}
.sig.none{{background:#161b26;color:#848e9c;border:1px solid #262e3d}}
.wrap{{background:#161b26;border:1px solid #262e3d;border-radius:10px;overflow:hidden}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;padding:8px 9px;background:#1c2331;color:#848e9c;font-weight:500}}
td{{padding:8px 9px;border-top:1px solid #222a38}}
.tag{{font-size:10px;padding:1px 4px;border-radius:3px;background:#1e293b;color:#38bdf8;margin-left:3px}}
</style></head><body>
<h1>📈 Sub-B 量化模拟盘</h1><div class="sub">起跑日 {LIVE_START} · 行情截至 {mkt_date} · 更新于 {bj_time}</div>
<div class="grid">
  <div class="card"><div class="sub">模拟总资产 (初始$10万)</div><div class="val {'pos' if tot_ret>=0 else 'neg'}">${tot_eq:,.0f}</div><div class="sub">累计 {tot_ret:+.1f}% | 回撤 {max_dd:.1f}%</div></div>
  <div class="card"><div class="sub">同期基准 ({LIVE_START}起)</div><div class="val" style="font-size:14px;margin-top:6px">QQQ: {qqq_ret:+.1f}%<br>SPY: {spy_ret:+.1f}%</div></div>
</div>
<div class="sec">⚡ 明日开盘指令</div>{sig_html}
<div class="sec">💼 当前模拟持仓 ({len(holdings)}/{MAX_SLOTS} 只 · 蓝标为总仓占比)</div>
<div class="wrap"><table><thead><tr><th>标的/占比</th><th>成本/现价</th><th>浮盈</th><th>动态止损线</th></tr></thead><tbody>{h_html or "<tr><td colspan='4' class='sub'>当前空仓（100% 现金待命）</td></tr>"}</tbody></table></div>
<div class="sec">🧾 实盘已平仓记录</div>
<div class="wrap"><table><thead><tr><th>标的/日期</th><th>买→卖</th><th>盈亏</th><th>离场原因</th></tr></thead><tbody>{l_html or "<tr><td colspan='4' class='sub'>暂无平仓记录</td></tr>"}</tbody></table></div>
<div class="sub" style="text-align:center;margin:18px 0">Sub-B: 双突破底座 · 14%波动率控制 · 黄金GLD/短债避险</div>
</body></html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ Sub-B 模拟盘初始化完成！当前持仓: {len(holdings)} 只 | 总资产: ${tot_eq:,.0f}")
