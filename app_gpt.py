import io
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =========================================================
# 1. 폰트 설정 (깃허브 fonts 폴더의 NanumGothic.ttf 강제 로드)
# =========================================================
def setup_fonts():
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc("font", family=font_name)
            plt.rc("axes", unicode_minus=False)
            pdfmetrics.registerFont(TTFont("ReportFont", font_path))
            return "ReportFont"
        except Exception:
            pass

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return "Helvetica"


PDF_FONT = setup_fonts()

COLOR_INK = "#172033"
COLOR_MUTED = "#6B7280"
COLOR_LINE = "#D8DEE9"
COLOR_ACCENT = "#2563EB"
COLOR_GREEN = "#15803D"
COLOR_GREEN_BG = "#F0FDF4"
COLOR_BLUE_BG = "#EFF6FF"

st.set_page_config(
    page_title="고압인버터 에너지 절감 시뮬레이터",
    layout="wide",
)


# =========================================================
# 2. 참조 데이터 테이블 및 계산 로직
# =========================================================
EXCEL_MOTOR_TABLE = [
    (0, 91.0), (55, 91.0), (90, 92.0), (132, 93.0),
    (160, 90.0), (400, 94.0), (630, 94.5), (800, 94.5),
    (1250, 96.0), (1800, 96.5), (3000, 97.0),
]

IE3_MOTOR_TABLE = [
    (0, 85.5), (55, 95.0), (90, 95.4), (132, 95.8),
    (160, 95.8), (375, 96.2), (630, 96.5), (800, 96.7),
    (1250, 97.1), (1800, 97.3), (3000, 97.5),
]

VALVE_RATIO_TABLE = [
    (53.5, 79.7571428571428), (60.2, 82.5589507464769),
    (66.8, 84.8751220873447), (75.5, 89.3400306962467),
    (80.2, 91.8515417887540), (83.4, 93.5956545730446),
    (93.6, 97.6698758197293), (100.0, 100.0),
]

N_MIN, N_MAX = 53.5, 100.0


def lookup_interp(x, table):
    return float(np.interp(x, [row[0] for row in table], [row[1] for row in table]))


def loss_profile_from_excel(pump_rated_shaft_kw):
    excel_eta_pct = lookup_interp(pump_rated_shaft_kw, EXCEL_MOTOR_TABLE)
    eta = excel_eta_pct / 100.0
    total_loss_kw = pump_rated_shaft_kw / eta - pump_rated_shaft_kw
    fixed_loss_fraction = pump_rated_shaft_kw * 0.0333 / total_loss_kw
    load_loss_fraction = 1.0 - fixed_loss_fraction
    return fixed_loss_fraction, load_loss_fraction, excel_eta_pct


def inverter_efficiency_pct(speed_ratio):
    if speed_ratio <= 0:
        return 0.0
    if speed_ratio <= 0.8:
        return 95.9
    return 98.1 - (0.5 + 0.6 * speed_ratio) / speed_ratio**3


@st.cache_data
def calculate_all(pump_rated_shaft_kw, flow_pct, mode, nameplate_efficiency_pct=None):
    selected_table = EXCEL_MOTOR_TABLE if mode == "excel" else IE3_MOTOR_TABLE
    reference_eta_pct = lookup_interp(pump_rated_shaft_kw, selected_table)
    applied_eta_pct = reference_eta_pct if nameplate_efficiency_pct is None else nameplate_efficiency_pct
    efficiency_source = "사내 엑셀 효율표" if nameplate_efficiency_pct is None else "현장 모터 명판 입력값"

    n = flow_pct / 100.0
    fixed_loss_fraction, load_loss_fraction, excel_eta_pct = loss_profile_from_excel(pump_rated_shaft_kw)
    total_loss_kw = pump_rated_shaft_kw / (applied_eta_pct / 100.0) - pump_rated_shaft_kw

    pm_inv = pump_rated_shaft_kw * n**3
    loss_inv_kw = (fixed_loss_fraction * n + load_loss_fraction * n**2) * total_loss_kw
    eta_motor_inv = pm_inv / (pm_inv + loss_inv_kw)
    eta_inv_pct = inverter_efficiency_pct(n)
    eta_total = eta_motor_inv * eta_inv_pct / 100.0
    p_inv_input = pm_inv / eta_total

    valve_ratio = lookup_interp(flow_pct, VALVE_RATIO_TABLE) / 100.0
    pm_valve = pump_rated_shaft_kw * valve_ratio
    loss_valve_kw = (fixed_loss_fraction + load_loss_fraction * valve_ratio**2) * total_loss_kw
    eta_motor_valve = pm_valve / (pm_valve + loss_valve_kw)
    p_valve_input = pm_valve / eta_motor_valve

    saving_kw = p_valve_input - p_inv_input
    return {
        "applied_eta_pct": applied_eta_pct,
        "efficiency_source": efficiency_source,
        "speed_ratio": n,
        "Pm_inv": pm_inv,
        "eta_motor_inv_pct": eta_motor_inv * 100.0,
        "eta_inv_self_pct": eta_inv_pct,
        "eta_total_pct": eta_total * 100.0,
        "P_inv_input": p_inv_input,
        "valve_ratio_pct": valve_ratio * 100.0,
        "Pm_valve": pm_valve,
        "P_valve_input": p_valve_input,
        "saving_kw": saving_kw,
        "saving_pct": saving_kw / p_valve_input * 100.0,
    }


def generate_figure(pump_kw, flow_pct, mode, nameplate_efficiency_pct, result):
    flows = np.linspace(N_MIN, N_MAX, 100)
    curve_inv, curve_valve = [], []
    for flow in flows:
        res_c = calculate_all(pump_kw, float(flow), mode, nameplate_efficiency_pct)
        curve_inv.append(res_c["P_inv_input"])
        curve_valve.append(res_c["P_valve_input"])

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(flows, curve_valve, color="#64748B", label="Valve 제어", linewidth=2.2)
    ax.plot(flows, curve_inv, color=COLOR_ACCENT, label="인버터 제어", linewidth=2.2)
    ax.fill_between(flows, curve_inv, curve_valve, color="#86EFAC", alpha=0.28, label="절감 구간")
    ax.scatter([flow_pct], [result["P_valve_input"]], color="#64748B", s=48, zorder=5)
    ax.scatter([flow_pct], [result["P_inv_input"]], color=COLOR_ACCENT, s=48, zorder=5)
    ax.axvline(x=flow_pct, color=COLOR_LINE, linestyle="--", linewidth=1.2)
    ax.set_xlabel("유량 / 속도 (%)")
    ax.set_ylabel("소비전력 (kW)")
    ax.set_title("유량 변화에 따른 소비전력")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# =========================================================
# 3. 사이드바 및 메인 화면 UI
# =========================================================
st.markdown("### 고압인버터 에너지 절감 시뮬레이션")

st.sidebar.header("운전 조건 설정")
mode_option = st.sidebar.radio("모터 효율 기준", ("사내 엑셀 기준", "IE3 참조 효율 기준"))
calc_mode = "excel" if mode_option == "사내 엑셀 기준" else "ie3"
in_pm = st.sidebar.number_input("Pump Data Shaft Power (kW)", value=724.9, min_value=1.0, step=10.0)
in_flow = st.sidebar.number_input("평균 운전 유량 또는 속도 (%)", value=75.5, min_value=N_MIN, max_value=N_MAX, step=0.1)
in_hours = st.sidebar.number_input("연간 운전 시간 (h)", value=8400, min_value=0, step=100)
in_price = st.sidebar.number_input("전기요금 (원/kWh)", value=120.4, min_value=0.0, step=0.1)
in_inv_cost = st.sidebar.number_input("인버터 도입 및 공사 비용 (원)", value=150000000.0, min_value=0.0, step=1000000.0)

res = calculate_all(in_pm, in_flow, calc_mode)
saving_won = res["saving_kw"] * in_hours * in_price
payback_years = (in_inv_cost / saving_won) if saving_won > 0 else None

fig = generate_figure(in_pm, in_flow, calc_mode, None, res)

st.write(f"**밸브 제어 소비전력:** {res['P_valve_input']:,.1f} kW")
st.write(f"**인버터 제어 소비전력:** {res['P_inv_input']:,.1f} kW")
st.write(f"**연간 절감 금액:** {saving_won / 1e8:.2f} 억원")
st.write(f"**예상 투자 회수기간:** {payback_years:.2f} 년" if payback_years else "산출 불가")

st.pyplot(fig)
