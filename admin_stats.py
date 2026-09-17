"""Private Streamlit dashboard for the ESRD application's usage totals."""

import logging
import re
from datetime import datetime, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

import requests
import streamlit as st


_LOG = logging.getLogger("esrd_usage_dashboard")


def _d1_config():
    """Return a validated, fixed-host D1 endpoint and its server-side token."""
    config = st.secrets["usage_tracking"]
    account_id = str(config["cloudflare_account_id"]).strip()
    database_id = str(config["d1_database_id"]).strip()
    token = str(config["cloudflare_api_token"]).strip()

    if not re.fullmatch(r"[0-9a-fA-F]{32}", account_id):
        raise ValueError("Invalid Cloudflare account ID")
    database_id = str(UUID(database_id))
    if not token or any(character.isspace() for character in token):
        raise ValueError("Invalid Cloudflare API token")

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + account_id
        + "/d1/database/"
        + database_id
        + "/query"
    )
    return endpoint, token


def _load_totals():
    endpoint, token = _d1_config()
    with requests.post(
        endpoint,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        json={
            "sql": (
                "SELECT visits, predict_clicks "
                "FROM esrd_usage_totals WHERE id = 1;"
            )
        },
        timeout=(2.0, 5.0),
        allow_redirects=False,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError("D1 request failed")
        body = response.json()

    if not isinstance(body, dict) or body.get("success") is not True or body.get("errors"):
        raise RuntimeError("D1 did not confirm the query")
    results = body.get("result")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Unexpected D1 result")
    query_result = results[0]
    if not isinstance(query_result, dict) or query_result.get("success") is not True:
        raise RuntimeError("D1 query failed")
    rows = query_result.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("Usage totals are missing")

    visits = rows[0].get("visits")
    predict_clicks = rows[0].get("predict_clicks")
    if (
        not isinstance(visits, int)
        or isinstance(visits, bool)
        or visits < 0
        or not isinstance(predict_clicks, int)
        or isinstance(predict_clicks, bool)
        or predict_clicks < 0
    ):
        raise RuntimeError("Usage totals are invalid")
    return visits, predict_clicks


st.set_page_config(
    page_title="ESRD Prediction Usage Statistics",
    page_icon="📊",
    layout="centered",
)

st.title("肾衰竭预测系统使用统计")
st.caption("累计统计 · 数据来自 Cloudflare D1")

if st.button("刷新数据", type="primary", use_container_width=True):
    pass

try:
    total_visits, total_predict_clicks = _load_totals()
except Exception as error:
    _LOG.warning("Usage dashboard query failed (%s).", type(error).__name__)
    st.error("统计数据暂时无法读取，请稍后刷新或联系管理员。")
    st.stop()

visits_column, clicks_column = st.columns(2)
with visits_column:
    st.metric("累计访问次数", f"{total_visits:,}")
with clicks_column:
    st.metric("累计 PREDICT 点击次数", f"{total_predict_clicks:,}")

updated_at = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime(
    "%Y-%m-%d %H:%M:%S %Z"
)
st.caption(f"页面读取时间：{updated_at}")
st.info("本页面仅展示累计使用次数，不包含患者输入、预测结果或身份信息。")
