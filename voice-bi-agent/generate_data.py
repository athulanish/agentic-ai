"""Generate a synthetic multi-region retail dataset with planted business stories."""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

rng = np.random.default_rng(42)

# ---------- dimensions ----------
products = pd.DataFrame([
    # product_id, name, category, list_price, unit_cost
    ("P001", "AeroBook 14", "Laptops", 980, 700),
    ("P002", "AeroBook Pro 16", "Laptops", 1850, 1350),
    ("P003", "Nimbus Phone S", "Phones", 720, 480),
    ("P004", "Nimbus Phone Max", "Phones", 1090, 740),
    ("P005", "SoundPod Lite", "Audio", 95, 48),
    ("P006", "SoundPod Studio", "Audio", 310, 170),
    ("P007", "HomeHub Mini", "Smart Home", 85, 40),
    ("P008", "HomeHub Cam 360", "Smart Home", 140, 72),
    ("P009", "HomeHub Thermostat", "Smart Home", 210, 105),
    ("P010", "ChargeFast 65W", "Accessories", 45, 16),
    ("P011", "ProCase Sleeve", "Accessories", 38, 12),
    ("P012", "DockPro 9-in-1", "Accessories", 120, 55),
], columns=["product_id", "product_name", "category", "list_price_usd", "unit_cost_usd"])

regions = {  # region: (country, weight, base_multiplier_for_price)
    "Dubai": ("UAE", 0.26, 1.00),
    "Abu Dhabi": ("UAE", 0.14, 1.00),
    "Riyadh": ("KSA", 0.22, 1.03),
    "Jeddah": ("KSA", 0.12, 1.03),
    "Kochi": ("India", 0.10, 0.92),
    "Bengaluru": ("India", 0.16, 0.92),
}
channels = {"Online": 0.46, "Retail Store": 0.40, "B2B": 0.14}
segments_by_channel = {
    "Online": (["Consumer", "SMB"], [0.9, 0.1]),
    "Retail Store": (["Consumer", "SMB"], [0.85, 0.15]),
    "B2B": (["SMB", "Enterprise"], [0.55, 0.45]),
}
cat_weights = {"Laptops": 0.14, "Phones": 0.22, "Audio": 0.16, "Smart Home": 0.20, "Accessories": 0.28}

# ---------- order generation ----------
dates = pd.date_range("2025-01-01", "2026-09-30", freq="D")

def seasonal(d):
    m = d.month
    s = {1: 0.95, 2: 0.95, 3: 1.05, 4: 0.95, 5: 0.95, 6: 0.85, 7: 0.8,
         8: 0.9, 9: 1.0, 10: 1.05, 11: 1.45, 12: 1.25}[m]
    growth = 1 + 0.012 * ((d.year - 2025) * 12 + m - 1)      # steady growth
    weekend = 1.15 if d.dayofweek in (4, 5) else 1.0          # Fri/Sat peak in the Gulf
    return s * growth * weekend

daily_orders = np.array([seasonal(d) for d in dates])
daily_orders = daily_orders / daily_orders.sum() * 9000
counts = rng.poisson(daily_orders)

rows = []
oid = 100000
for d, n in zip(dates, counts):
    for _ in range(n):
        oid += 1
        region = rng.choice(list(regions), p=[v[1] for v in regions.values()])
        channel = rng.choice(list(channels), p=list(channels.values()))
        segs, sp = segments_by_channel[channel]
        segment = rng.choice(segs, p=sp)
        cat = rng.choice(list(cat_weights), p=list(cat_weights.values()))
        prod = products[products.category == cat].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        if channel == "Online":
            hour = int(np.clip(rng.normal(21, 3), 0, 23))
        elif channel == "Retail Store":
            hour = int(np.clip(rng.normal(17, 3), 10, 22))
        else:
            hour = int(np.clip(rng.normal(11, 2), 8, 17))
        t = f"{hour:02d}:{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}"
        qty = int(rng.choice([1, 2, 3, 5, 10, 25], p=[.62, .2, .08, .05, .03, .02])) if channel == "B2B" else int(rng.choice([1, 2, 3], p=[.8, .15, .05]))
        disc = float(rng.choice([0, .05, .10, .15], p=[.55, .25, .15, .05]))
        if d.month == 11:
            disc = float(rng.choice([.10, .15, .20, .25], p=[.3, .3, .3, .1]))
        if channel == "B2B" and segment == "Enterprise":
            disc += 0.08
        # STORY 1: Dubai online discounting spree in Q2 2026 (margin squeeze)
        if region == "Dubai" and channel == "Online" and d >= pd.Timestamp("2026-04-01") and d <= pd.Timestamp("2026-06-30"):
            disc = float(rng.choice([.20, .25, .30], p=[.4, .4, .2]))
        price = round(prod.list_price_usd * regions[region][2], 2)
        revenue = round(price * qty * (1 - disc), 2)
        cogs = round(prod.unit_cost_usd * qty, 2)
        # returns
        p_ret = 0.04 if cat != "Smart Home" else 0.05
        # STORY 2: Riyadh Smart Home returns spike Jun-Aug 2026 (defective batch)
        if region == "Riyadh" and cat == "Smart Home" and pd.Timestamp("2026-06-01") <= d <= pd.Timestamp("2026-08-31"):
            p_ret = 0.22
        returned = bool(rng.random() < p_ret)
        base_days = {"Online": 3, "Retail Store": 0, "B2B": 6}[channel]
        delivery = 0 if channel == "Retail Store" else max(1, int(rng.normal(base_days, 1.2)))
        # STORY 3: Kochi delivery delays from Sep 2026 (logistics partner issue)
        if region == "Kochi" and channel != "Retail Store" and d >= pd.Timestamp("2026-08-15"):
            delivery += int(rng.integers(3, 6))
        csat = int(np.clip(round(rng.normal(4.3 - 0.12 * max(delivery - 4, 0) - (1.6 if returned else 0), 0.7)), 1, 5))
        rows.append((f"ORD{oid}", d.date(), t, region, regions[region][0], channel, segment,
                     prod.product_id, prod.product_name, cat, qty, price, disc,
                     revenue, cogs, returned, delivery, csat))

cols = ["order_id", "order_date", "order_time", "region", "country", "channel", "customer_segment",
        "product_id", "product_name", "category", "quantity", "unit_price_usd", "discount_pct",
        "revenue_usd", "cogs_usd", "returned", "delivery_days", "csat_score"]
orders = pd.DataFrame(rows, columns=cols)
orders["profit_usd"] = (orders.revenue_usd - orders.cogs_usd).round(2)
orders.loc[orders.returned, "profit_usd"] = (-orders.cogs_usd * 0.15).round(2)  # simplified return cost
orders["returned"] = orders["returned"].map({True: "Yes", False: "No"})

# ---------- monthly targets per region ----------
orders["month"] = pd.to_datetime(orders.order_date).dt.to_period("M")
actual = orders.groupby(["month", "region"]).revenue_usd.sum().reset_index()
actual["target_usd"] = (actual.revenue_usd * rng.uniform(0.93, 1.08, len(actual))).round(-2)
# STORY 4: Bengaluru consistently over-delivers target in 2026
mask = (actual.region == "Bengaluru") & (actual.month >= pd.Period("2026-01"))
actual.loc[mask, "target_usd"] = (actual.loc[mask, "revenue_usd"] * 0.9).round(-2)
targets = actual[["month", "region", "target_usd"]].copy()
targets["month"] = targets.month.dt.to_timestamp().dt.date
orders = orders.drop(columns="month")

dictionary = pd.DataFrame([
    ("Orders", "order_id", "Unique order id"),
    ("Orders", "order_date / order_time", "Order timestamp split into date and time (24h)"),
    ("Orders", "region / country", "Sales region and its country (Dubai, Abu Dhabi, Riyadh, Jeddah, Kochi, Bengaluru)"),
    ("Orders", "channel", "Online, Retail Store or B2B"),
    ("Orders", "customer_segment", "Consumer, SMB or Enterprise"),
    ("Orders", "discount_pct", "Discount as a fraction (0.10 = 10%)"),
    ("Orders", "revenue_usd", "unit_price * quantity * (1 - discount)"),
    ("Orders", "cogs_usd", "unit_cost * quantity"),
    ("Orders", "profit_usd", "revenue - cogs; returned orders carry a flat 15% of cogs loss instead"),
    ("Orders", "returned", "Yes if the order was returned"),
    ("Orders", "delivery_days", "Days to deliver (0 for in-store)"),
    ("Orders", "csat_score", "Customer satisfaction 1 to 5"),
    ("Products", "list_price_usd / unit_cost_usd", "Base price and cost before regional price index"),
    ("Targets", "target_usd", "Monthly revenue target per region"),
], columns=["sheet", "column", "description"])

# ---------- write workbook ----------
out = "data/business_data.xlsx"
with pd.ExcelWriter(out, engine="openpyxl", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
    orders.to_excel(xw, sheet_name="Orders", index=False)
    products.to_excel(xw, sheet_name="Products", index=False)
    targets.to_excel(xw, sheet_name="Targets", index=False)
    dictionary.to_excel(xw, sheet_name="Data_Dictionary", index=False)

from openpyxl import load_workbook
wb = load_workbook(out)
hdr_fill = PatternFill("solid", fgColor="1F3864")
for ws in wb.worksheets:
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
    for i, col in enumerate(ws.columns, 1):
        w = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
        ws.column_dimensions[get_column_letter(i)].width = min(max(w + 3, 12), 48)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

# Summary sheet with live formulas as a sanity check
s = wb.create_sheet("Summary", 0)
n = len(orders) + 1
s["A1"] = "Sanity-check summary (live formulas over the Orders sheet)"
s["A1"].font = Font(name="Arial", bold=True, size=12)
heads = ["Region", "Orders", "Revenue (USD)", "Profit (USD)", "Margin %", "Return rate %"]
for j, h in enumerate(heads, 1):
    c = s.cell(row=3, column=j, value=h)
    c.font = Font(name="Arial", bold=True, color="FFFFFF"); c.fill = hdr_fill
for i, r in enumerate(regions, 4):
    s.cell(row=i, column=1, value=r)
    s.cell(row=i, column=2, value=f'=COUNTIF(Orders!$D$2:$D${n},A{i})')
    s.cell(row=i, column=3, value=f'=SUMIFS(Orders!$N$2:$N${n},Orders!$D$2:$D${n},A{i})')
    s.cell(row=i, column=4, value=f'=SUMIFS(Orders!$S$2:$S${n},Orders!$D$2:$D${n},A{i})')
    s.cell(row=i, column=5, value=f'=IF(C{i}=0,0,D{i}/C{i})')
    s.cell(row=i, column=6, value=f'=IF(B{i}=0,0,COUNTIFS(Orders!$D$2:$D${n},A{i},Orders!$P$2:$P${n},"Yes")/B{i})')
t = 4 + len(regions)
s.cell(row=t, column=1, value="Total")
for col in "BCD":
    s[f"{col}{t}"] = f"=SUM({col}4:{col}{t-1})"
s[f"E{t}"] = f"=IF(C{t}=0,0,D{t}/C{t})"
s[f"F{t}"] = f'=COUNTIF(Orders!$P$2:$P${n},"Yes")/B{t}'
for r in s.iter_rows(min_row=4, max_row=t):
    for c in r:
        c.font = Font(name="Arial", size=10, bold=(c.row == t))
    r[1].number_format = "#,##0"; r[2].number_format = "#,##0"; r[3].number_format = "#,##0"
    r[4].number_format = "0.0%"; r[5].number_format = "0.0%"
for col, w in zip("ABCDEF", [16, 12, 18, 18, 12, 16]):
    s.column_dimensions[col].width = w
wb.save(out)
print(f"{len(orders):,} orders written to {out}")
