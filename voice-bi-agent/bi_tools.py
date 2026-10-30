"""Deterministic analytics tools the agent can call. The LLM never writes code or does arithmetic;
it only picks a tool and arguments, and pandas does the maths."""
import pandas as pd

DATA_PATH = "data/business_data.xlsx"

DIMENSIONS = ["region", "country", "channel", "customer_segment", "category",
              "product_name", "month", "quarter", "weekday", "hour"]
FILTERABLE = ["region", "country", "channel", "customer_segment", "category", "product_name", "returned"]
METRICS = ["revenue", "profit", "orders", "units", "margin_pct", "return_rate_pct",
           "avg_order_value", "avg_discount_pct", "avg_delivery_days", "avg_csat"]


def load(path: str = DATA_PATH):
    orders = pd.read_excel(path, sheet_name="Orders")
    targets = pd.read_excel(path, sheet_name="Targets")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["month"] = orders["order_date"].dt.to_period("M").astype(str)
    orders["quarter"] = orders["order_date"].dt.to_period("Q").astype(str)
    orders["weekday"] = orders["order_date"].dt.day_name()
    orders["hour"] = orders["order_time"].astype(str).str[:2].astype(int)
    # in-store orders have no delivery, so keep them out of delivery averages
    orders["delivery_days_nz"] = orders["delivery_days"].where(orders["channel"] != "Retail Store")
    targets["month"] = pd.to_datetime(targets["month"]).dt.to_period("M").astype(str)
    return orders, targets


ORDERS, TARGETS = load()


def _slice(start_date=None, end_date=None, filters=None):
    df = ORDERS
    if start_date:
        df = df[df.order_date >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.order_date <= pd.Timestamp(end_date)]
    for col, val in (filters or {}).items():
        if col not in FILTERABLE:
            raise ValueError(f"Cannot filter on '{col}'. Allowed: {FILTERABLE}")
        vals = val if isinstance(val, list) else [val]
        df = df[df[col].isin(vals)]
    return df


def _metrics_frame(df, group_by):
    g = df.groupby(group_by) if group_by else df.groupby(lambda _: "All")
    out = g.agg(revenue=("revenue_usd", "sum"), profit=("profit_usd", "sum"),
                orders=("order_id", "count"), units=("quantity", "sum"),
                returned_n=("returned", lambda s: (s == "Yes").sum()),
                avg_discount_pct=("discount_pct", "mean"),
                avg_delivery_days=("delivery_days_nz", "mean"),
                avg_csat=("csat_score", "mean"))
    out["margin_pct"] = out.profit / out.revenue * 100
    out["return_rate_pct"] = out.returned_n / out.orders * 100
    out["avg_order_value"] = out.revenue / out.orders
    out["avg_discount_pct"] = out.avg_discount_pct * 100
    return out


def _check(metric, group_by):
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {METRICS}")
    if group_by and group_by not in DIMENSIONS:
        raise ValueError(f"Unknown group_by '{group_by}'. Allowed: {DIMENSIONS}")


def describe_data():
    return {
        "date_range": [str(ORDERS.order_date.min().date()), str(ORDERS.order_date.max().date())],
        "rows": len(ORDERS),
        "currency": "USD",
        "metrics": METRICS,
        "group_by_options": DIMENSIONS,
        "filter_values": {c: sorted(ORDERS[c].unique().tolist()) for c in
                          ["region", "country", "channel", "customer_segment", "category"]},
    }


def query_metrics(metric, group_by=None, filters=None, start_date=None, end_date=None,
                  sort_desc=True, limit=10):
    _check(metric, group_by)
    df = _slice(start_date, end_date, filters)
    if df.empty:
        return {"rows": [], "note": "No data for this slice"}
    res = _metrics_frame(df, group_by)[metric]
    if group_by in ("month", "quarter", "weekday", "hour"):
        res = res.sort_index() if group_by != "weekday" else res.reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).dropna()
    else:
        res = res.sort_values(ascending=not sort_desc)
    res = res.head(limit) if group_by not in ("month", "quarter", "weekday", "hour") else res
    return {"metric": metric, "group_by": group_by,
            "rows": [{"group": str(k), "value": round(float(v), 2)} for k, v in res.items()]}


def compare_periods(metric, a_start, a_end, b_start, b_end, group_by=None, filters=None):
    """Period A vs period B (B is usually the earlier baseline). Returns change of A relative to B."""
    _check(metric, group_by)
    a = _metrics_frame(_slice(a_start, a_end, filters), group_by)[metric]
    b = _metrics_frame(_slice(b_start, b_end, filters), group_by)[metric]
    both = pd.concat([a.rename("period_a"), b.rename("period_b")], axis=1).fillna(0)
    both["change"] = both.period_a - both.period_b
    both["change_pct"] = both.apply(lambda r: r.change / r.period_b * 100 if r.period_b else None, axis=1)
    both = both.sort_values("change")
    return {"metric": metric, "group_by": group_by, "rows": [
        {"group": str(k), "period_a": round(r.period_a, 2), "period_b": round(r.period_b, 2),
         "change": round(r.change, 2),
         "change_pct": None if pd.isna(r.change_pct) else round(r.change_pct, 1)}
        for k, r in both.iterrows()]}


def vs_target(start_month, end_month, region=None):
    """Revenue vs target by region for a month range (YYYY-MM)."""
    t = TARGETS[(TARGETS.month >= start_month) & (TARGETS.month <= end_month)]
    o = ORDERS[(ORDERS.month >= start_month) & (ORDERS.month <= end_month)]
    if region:
        t, o = t[t.region == region], o[o.region == region]
    res = pd.concat([o.groupby("region").revenue_usd.sum().rename("actual"),
                     t.groupby("region").target_usd.sum().rename("target")], axis=1)
    res["attainment_pct"] = res.actual / res.target * 100
    return {"rows": [{"region": k, "actual": round(r.actual, 0), "target": round(r.target, 0),
                      "attainment_pct": round(r.attainment_pct, 1)} for k, r in res.iterrows()]}


# ---- OpenAI tool schemas ----
_FILTERS = {"type": "object", "description": "Column to value or list of values. Keys: " + ", ".join(FILTERABLE),
            "additionalProperties": True}
SCHEMAS = [
    {"type": "function", "function": {
        "name": "describe_data", "description": "Date range, valid metrics, dimensions and filter values.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "query_metrics",
        "description": "One metric, optionally grouped by a dimension (use month/quarter for trends), with filters and a date range.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": METRICS},
            "group_by": {"type": "string", "enum": DIMENSIONS},
            "filters": _FILTERS,
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "sort_desc": {"type": "boolean"}, "limit": {"type": "integer"}},
            "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "compare_periods",
        "description": "Compare a metric between period A and baseline period B, optionally per group. Use for 'what changed', 'vs last quarter', 'year on year'.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": METRICS},
            "a_start": {"type": "string"}, "a_end": {"type": "string"},
            "b_start": {"type": "string"}, "b_end": {"type": "string"},
            "group_by": {"type": "string", "enum": DIMENSIONS}, "filters": _FILTERS},
            "required": ["metric", "a_start", "a_end", "b_start", "b_end"]}}},
    {"type": "function", "function": {
        "name": "vs_target",
        "description": "Revenue versus target by region over a month range.",
        "parameters": {"type": "object", "properties": {
            "start_month": {"type": "string", "description": "YYYY-MM"},
            "end_month": {"type": "string", "description": "YYYY-MM"},
            "region": {"type": "string"}},
            "required": ["start_month", "end_month"]}}},
]
REGISTRY = {"describe_data": describe_data, "query_metrics": query_metrics,
            "compare_periods": compare_periods, "vs_target": vs_target}
