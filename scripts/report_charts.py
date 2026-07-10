"""HTML 리포트 전용 렌더링 (matplotlib 정적 이미지 + HTML 테이블).

메인 대시보드(src/charts.py)는 Plotly 인터랙티브 차트를 쓰지만, 정적 HTML 리포트를
브라우저에서 열 때 무겁고 느려지는 문제가 있어 리포트는 여기서 별도로 렌더링한다.

- 차트류(NAV, Drawdown, Annual Returns, Holdings Heatmap): matplotlib → base64 PNG <img>
- 표류(Signal Category, Monthly Returns): 순수 HTML <table>

계산 로직은 전혀 포함하지 않는다. generate_report.py가 넘겨주는, 이미 계산이 끝난
DataFrame/Series를 받아 표현만 담당한다.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")  # 헤드리스 백엔드 (cron/서버에서 디스플레이 불필요)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Plotly qualitative Set2 팔레트를 hex로 복제 (대시보드와 외형 근접)
SET2 = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC0",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]


def _fig_to_img(fig) -> str:
    """matplotlib Figure를 base64 PNG <img> 조각으로 변환하고 figure를 닫는다."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;height:auto;" alt="chart">'
    )


def nav_chart_img(navs: pd.DataFrame, title: str = "Portfolio NAV") -> str:
    """NAV 시계열 라인 차트."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, col in enumerate(navs.columns):
        ax.plot(navs.index, navs[col], color=SET2[i % len(SET2)],
                linewidth=2, label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    return _fig_to_img(fig)


def drawdown_chart_img(navs: pd.DataFrame, title: str = "Drawdown (%)") -> str:
    """Drawdown 영역 차트."""
    cummax = navs.cummax()
    dd = (navs - cummax) / cummax * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, col in enumerate(dd.columns):
        color = SET2[i % len(SET2)]
        ax.fill_between(dd.index, dd[col], 0, color=color, alpha=0.35)
        ax.plot(dd.index, dd[col], color=color, linewidth=1, label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    return _fig_to_img(fig)


def annual_returns_img(annual_rets: pd.DataFrame, title: str = "Annual Returns") -> str:
    """연도별 수익률 그룹 막대 차트."""
    years = [str(y) for y in annual_rets.index]
    n_series = len(annual_rets.columns)
    x = np.arange(len(years))
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, col in enumerate(annual_rets.columns):
        vals = annual_rets[col].values
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=col, color=SET2[i % len(SET2)])
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.1%}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
            )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=9)
    return _fig_to_img(fig)


def holding_heatmap_img(weight: pd.DataFrame, title: str = "Monthly Holdings") -> str:
    """월별 보유 히트맵. 보유하지 않은(가중치 0) 티커도 유니버스 전체를 y축에 표시."""
    w = weight.copy()
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()
    weight_t = w.T  # 행: 티커(유니버스 전체), 열: 월

    tickers = list(weight_t.index)
    x_dates = pd.to_datetime(weight_t.columns)
    z = weight_t.values

    fig_h = max(5.0, len(tickers) * 0.28)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    im = ax.imshow(z, aspect="auto", cmap="Blues", vmin=0)

    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    ax.set_xticks(range(len(x_dates)))
    ax.set_xticklabels(
        [d.strftime("%y.%m") for d in x_dates], rotation=45, ha="right", fontsize=8
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Ticker")
    if title:
        ax.set_title(title)
    return _fig_to_img(fig)


def signal_category_html(
    signal: pd.Series, n_quantiles: int = 5
) -> str:
    """현재 시그널을 분위(Q5=최고 ~ Q1=최저)별로 묶은 HTML 테이블.

    src.charts.create_signal_category_table의 그룹화/색상 로직을 그대로 옮겨 렌더만 HTML로.
    """
    signal = signal.dropna()

    # 분위 배정: 최고 시그널 -> Q5, 최저 -> Q1 (동점은 평균 백분위 + 고정 빈으로 유지)
    pct = signal.rank(pct=True)
    bins = np.linspace(0, 1, n_quantiles + 1)
    quantile_labels = pd.cut(
        pct,
        bins=bins,
        labels=[f"Q{i}" for i in range(n_quantiles, 0, -1)],
        include_lowest=True,
    )

    quantile_data = {}
    for q in range(n_quantiles, 0, -1):
        q_label = f"Q{q}"
        tickers_in_q = quantile_labels[quantile_labels == q_label].index.tolist()
        tickers_sorted = signal.loc[tickers_in_q].sort_values()
        quantile_data[q_label] = [
            f"{t} ({int(signal[t])})" for t in tickers_sorted.index
        ]

    max_len = max((len(v) for v in quantile_data.values()), default=0)
    for q in quantile_data:
        while len(quantile_data[q]) < max_len:
            quantile_data[q].append("")

    headers = [f"Q{q}" for q in range(n_quantiles, 0, -1)]

    # 헤더 색상: Q5(최고)=초록, Q1(최저)=빨강
    header_colors = []
    for q in range(n_quantiles, 0, -1):
        ratio = (q - 1) / (n_quantiles - 1) if n_quantiles > 1 else 1.0
        r = int(255 * (1 - ratio))
        g = int(200 * ratio)
        header_colors.append(f"rgb({r}, {g}, 100)")

    ths = "".join(
        f'<th style="background:{c};color:white;">{h}</th>'
        for h, c in zip(headers, header_colors)
    )
    rows = ""
    for i in range(max_len):
        cells = "".join(f"<td>{quantile_data[h][i]}</td>" for h in headers)
        rows += f"<tr>{cells}</tr>"

    return (
        '<table class="data-table signal-table">'
        f"<thead><tr>{ths}</tr></thead><tbody>{rows}</tbody></table>"
    )


def monthly_returns_html(monthly_rets: pd.DataFrame) -> str:
    """월별 수익률 (연 x 월) HTML 테이블. 셀 배경은 RdYlGn 컬러맵(0 대칭)."""
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    vals = monthly_rets.values
    finite = vals[np.isfinite(vals)]
    absmax = float(np.max(np.abs(finite))) if finite.size else 0.0
    if absmax <= 0:
        absmax = 1e-9
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=-absmax, vmax=absmax)

    header = "".join(f"<th>{m}</th>" for m in month_labels)
    rows = ""
    for year, row in monthly_rets.iterrows():
        cells = ""
        for v in row.values:
            if pd.isna(v):
                cells += "<td></td>"
            else:
                bg = mcolors.to_hex(cmap(norm(v)))
                cells += f'<td style="background:{bg};">{v * 100:.1f}%</td>'
        rows += f"<tr><th>{year}</th>{cells}</tr>"

    return (
        '<table class="data-table monthly-table">'
        f"<thead><tr><th>Year</th>{header}</tr></thead><tbody>{rows}</tbody></table>"
    )
