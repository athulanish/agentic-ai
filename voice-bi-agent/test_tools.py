"""No-API sanity tests: tool outputs must match straight pandas on the raw Excel sheet."""
import pandas as pd
import bi_tools as t

raw = pd.read_excel(t.DATA_PATH, sheet_name="Orders")
raw["order_date"] = pd.to_datetime(raw.order_date)

def close(a, b, tol=0.01): assert abs(a - b) <= tol, (a, b)

# 1. total revenue by region
res = {r["group"]: r["value"] for r in t.query_metrics("revenue", "region")["rows"]}
for k, v in raw.groupby("region").revenue_usd.sum().items(): close(res[k], v)

# 2. margin for Dubai online in Q2 2026
r = t.query_metrics("margin_pct", filters={"region": "Dubai", "channel": "Online"},
                    start_date="2026-04-01", end_date="2026-06-30")["rows"][0]["value"]
s = raw[(raw.region == "Dubai") & (raw.channel == "Online") & raw.order_date.between("2026-04-01", "2026-06-30")]
close(r, (s.profit_usd.sum() / s.revenue_usd.sum()) * 100)

# 3. return rate, Riyadh smart home Jul-Aug 2026
r = t.query_metrics("return_rate_pct", filters={"region": "Riyadh", "category": "Smart Home"},
                    start_date="2026-07-01", end_date="2026-08-31")["rows"][0]["value"]
s = raw[(raw.region == "Riyadh") & (raw.category == "Smart Home") & raw.order_date.between("2026-07-01", "2026-08-31")]
close(r, (s.returned == "Yes").mean() * 100)

# 4. compare periods: Q3 2026 vs Q2 2026 revenue
c = t.compare_periods("revenue", "2026-07-01", "2026-09-30", "2026-04-01", "2026-06-30")["rows"][0]
a = raw[raw.order_date.between("2026-07-01", "2026-09-30")].revenue_usd.sum()
b = raw[raw.order_date.between("2026-04-01", "2026-06-30")].revenue_usd.sum()
close(c["period_a"], a); close(c["period_b"], b); close(c["change_pct"], (a - b) / b * 100, 0.1)

# 5. target attainment
v = {r["region"]: r for r in t.vs_target("2026-01", "2026-09")["rows"]}
assert v["Bengaluru"]["attainment_pct"] > 100, v["Bengaluru"]

# 6. bad inputs fail loudly, not silently
for bad in (lambda: t.query_metrics("vibes"), lambda: t.query_metrics("revenue", filters={"salary": 1})):
    try: bad(); raise SystemExit("expected an error")
    except ValueError: pass
print("all tool tests passed")
