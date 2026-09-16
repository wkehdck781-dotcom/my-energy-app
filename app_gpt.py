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
# 1. 폰트 및 서버 환경 대응 설정 (한글 깨짐 방지)
# =========================================================
def setup_fonts():
    # 1. 로컬에 폰트 파일이 있는 경우 우선 등록 시도
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

    # 2. 리눅스 서버(깃허브 등) 환경에서 시스템 나눔고딕 탐색 및 강제 지정
    if os.name == "posix":
        for candidate in ["NanumGothic", "Nanum Gothic", "DejaVu Sans"]:
            try:
                plt.rcParams["font.family"] = candidate
                break
            except Exception:
                pass
    else:
        plt.rcParams["font.family"] = "Malgun Gothic"

    plt.rcParams["axes.unicode_minus"] = False

    # 3. ReportLab PDF용 폰트 설정
    installed = {font.name: font.fname for font in fm.fontManager.ttflist}
    selected_path = None
    for candidate in ("NanumGothic", "Nanum Gothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"):
        if candidate in installed:
            selected_path = installed[candidate]
            break

    if selected_path and selected_path.lower().endswith((".ttf", ".ttc")):
        try:
            pdfmetrics.registerFont(TTFont("ReportFont", selected_path))
            return "ReportFont"
        except Exception:
            pass
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

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "Malgun Gothic", "NanumGothic", sans-serif;
    }}
    .app-title {{
        font-size: 26px; font-weight: 750; color: {COLOR_INK}; margin-bottom: 4px;
    }}
    .app-subtitle {{
        color: {COLOR_MUTED}; font-size: 14px; margin-bottom: 24px;
    }}
    .summary-grid {{
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px; margin: 8px 0 22px 0;
    }}
    .summary-card {{
        border: 1px solid {COLOR_LINE}; border-radius: 10px; padding: 18px 16px;
        background: white; min-height: 112px;
    }}
    .summary-card-blue {{ background: {COLOR_BLUE_BG}; border-color: #BFDBFE; }}
    .summary-card-green {{ background: {COLOR_GREEN_BG}; border-color: #BBF7D0; }}
    .summary-label {{ color: {COLOR_MUTED}; font-size: 13px; margin-bottom: 10px; }}
    .summary-value {{ color: {COLOR_INK}; font-size: 23px; font-weight: 750; line-height: 1.1; }}
    .summary-value-green {{ color: {COLOR_GREEN}; }}
    .summary-note {{ color: {COLOR_MUTED}; font-size: 12px; margin-top: 8px; }}
    @media (max-width: 900px) {{
        .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. 참조 데이터 테이블
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


# =========================================================
# 3. 계산 함수
# =========================================================
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
    if pump_rated_shaft_kw <= 0:
        raise ValueError("Pump Data Shaft Power는 0보다 커야 합니다.")
    if not N_MIN <= flow_pct <= N_MAX:
        raise ValueError(f"유량/속도는 {N_MIN}~{N_MAX}% 범위여야 합니다.")

    selected_table = EXCEL_MOTOR_TABLE if mode == "excel" else IE3_MOTOR_TABLE
    reference_eta_pct = lookup_interp(pump_rated_shaft_kw, selected_table)

    if nameplate_efficiency_pct is None:
        applied_eta_pct = reference_eta_pct
        efficiency_source = "사내 엑셀 효율표" if mode == "excel" else "IE3 참조 효율표"
    else:
        applied_eta_pct = nameplate_efficiency_pct
        efficiency_source = "현장 모터 명판 입력값"

    n = flow_pct / 100.0
    fixed_loss_fraction, load_loss_fraction, excel_eta_pct = loss_profile_from_excel(
        pump_rated_shaft_kw
    )

    total_loss_kw = pump_rated_shaft_kw / (applied_eta_pct / 100.0) - pump_rated_shaft_kw

    pm_inv = pump_rated_shaft_kw * n**3
    loss_inv_kw = (fixed_loss_fraction * n + load_loss_fraction * n**2) * total_loss_kw
    eta_motor_inv = pm_inv / (pm_inv + loss_inv_kw)
    eta_inv_pct = inverter_efficiency_pct(n)
    eta_total = eta_motor_inv * eta_inv_pct / 100.0
    p_inv_input = pm_inv / eta_total

    valve_ratio = lookup_interp(flow_pct, VALVE_RATIO_TABLE) / 100.0
    pm_valve = pump_rated_shaft_kw * valve_ratio
    loss_valve_kw = (
        fixed_loss_fraction + load_loss_fraction * valve_ratio**2
    ) * total_loss_kw
    eta_motor_valve = pm_valve / (pm_valve + loss_valve_kw)
    p_valve_input = pm_valve / eta_motor_valve

    saving_kw = p_valve_input - p_inv_input
    return {
        "mode": mode,
        "efficiency_source": efficiency_source,
        "reference_eta_pct": reference_eta_pct,
        "applied_eta_pct": applied_eta_pct,
        "excel_eta_pct": excel_eta_pct,
        "speed_ratio": n,
        "total_loss_kw": total_loss_kw,
        "fixed_loss_fraction": fixed_loss_fraction,
        "load_loss_fraction": load_loss_fraction,
        "Pm_inv": pm_inv,
        "loss_inv_kw": loss_inv_kw,
        "eta_motor_inv_pct": eta_motor_inv * 100.0,
        "eta_inv_self_pct": eta_inv_pct,
        "eta_total_pct": eta_total * 100.0,
        "P_inv_input": p_inv_input,
        "valve_ratio_pct": valve_ratio * 100.0,
        "Pm_valve": pm_valve,
        "loss_valve_kw": loss_valve_kw,
        "eta_motor_valve_pct": eta_motor_valve * 100.0,
        "P_valve_input": p_valve_input,
        "saving_kw": saving_kw,
        "saving_pct": saving_kw / p_valve_input * 100.0,
    }


def fmt(value, digits=1):
    return "-" if value is None else f"{value:,.{digits}f}"


def generate_figure(pump_kw, flow_pct, mode, nameplate_efficiency_pct, result):
    flows = np.linspace(N_MIN, N_MAX, 100)
    curve_inv, curve_valve = [], []

    for flow in flows:
        curve_result = calculate_all(pump_kw, float(flow), mode, nameplate_efficiency_pct)
        curve_inv.append(curve_result["P_inv_input"])
        curve_valve.append(curve_result["P_valve_input"])

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    
    ax.plot(flows, curve_valve, color="#94A3B8", label="Valve Control (Valve 제어)", linewidth=2.5)
    ax.plot(flows, curve_inv, color=COLOR_ACCENT, label="Inverter Control (인버터 제어)", linewidth=3.0)
    ax.fill_between(flows, curve_inv, curve_valve, color="#38BDF8", alpha=0.2, label="Energy Saving Area (절감 구간)")
    
    ax.scatter([flow_pct], [result["P_valve_input"]], color="#64748B", s=60, zorder=5, edgecolors="white", linewidths=1.5)
    ax.scatter([flow_pct], [result["P_inv_input"]], color=COLOR_ACCENT, s=70, zorder=5, edgecolors="white", linewidths=1.5)
    ax.axvline(x=flow_pct, color="#CBD5E1", linestyle=":", linewidth=1.5)
    
    # 폰트 깨짐 방지를 위해 범례 및 레이블 명시적 지정
    ax.set_xlabel("Flow / Speed Ratio (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Power Consumption (kW)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Power Consumption by Flow Rate (유량 변화에 따른 소비전력)", fontsize=12, fontweight="bold", pad=12)
    
    ax.grid(True, linestyle="--", alpha=0.4, color="#E2E8F0")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9)
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#94A3B8")
        
    plt.tight_layout()
    return fig


def generate_pdf_report(result, mode_name, pump_kw, flow_pct, hours, price, inv_cost, payback_years, figure):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=35, leftMargin=35, topMargin=35, bottomMargin=35
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PdfTitle", parent=styles["Heading1"], fontName=PDF_FONT,
                                  fontSize=15, leading=19, textColor=colors.HexColor(COLOR_INK))
    normal_style = ParagraphStyle("PdfNormal", parent=styles["Normal"], fontName=PDF_FONT,
                                  fontSize=8.5, leading=12, textColor=colors.HexColor(COLOR_INK))
    header_style = ParagraphStyle("PdfHeader", parent=styles["Normal"], fontName=PDF_FONT,
                                  fontSize=8.5, leading=11, alignment=1, textColor=colors.white)
    cell_style = ParagraphStyle("PdfCell", parent=styles["Normal"], fontName=PDF_FONT,
                                fontSize=8.5, leading=11, alignment=1, textColor=colors.HexColor(COLOR_INK))
    note_style = ParagraphStyle("PdfNote", parent=styles["Normal"], fontName=PDF_FONT,
                                fontSize=7.2, leading=10, textColor=colors.HexColor(COLOR_MUTED))

    def report_table(rows, widths):
        prepared = []
        for row_index, row in enumerate(rows):
            style = header_style if row_index == 0 else cell_style
            prepared.append([Paragraph(str(value), style) for value in row])
        table = Table(prepared, colWidths=widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_INK)),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(COLOR_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return table

    saving_won = result["saving_kw"] * hours * price
    payback_str = f"{payback_years:.2f} 년" if payback_years is not None else "산출 불가 (절감액 없음)"

    items = [
        Paragraph("고압인버터 에너지 절감 시뮬레이션 보고서", title_style),
        Spacer(1, 5),
        Paragraph(
            f"적용 기준: {mode_name} / 모터 효율: {result['efficiency_source']} "
            f"({result['applied_eta_pct']:.2f}%)<br/>"
            f"Pump Shaft Power: {pump_kw:.1f} kW / 운전 유량: {flow_pct:.1f}% / "
            f"연간 운전시간: {hours:,.0f} h / 전력단가: {price:,.1f} 원/kWh",
            normal_style,
        ),
        Spacer(1, 14),
        Paragraph("1. 핵심 절감 및 투자 회수 요약", title_style),
        Spacer(1, 6),
        report_table([
            ["항목", "Valve 제어", "인버터 제어", "절감 효과"],
            ["소비전력", f"{result['P_valve_input']:,.1f} kW", f"{result['P_inv_input']:,.1f} kW", f"{result['saving_kw']:,.1f} kW"],
            ["절감률", "-", "-", f"{result['saving_pct']:.1f}%"],
            ["연간 절감 금액", "-", "-", f"{saving_won / 1e8:.2f} 억원/년"],
            ["예상 투자 회수기간", "-", "-", payback_str],
        ], [115, 115, 115, 145]),
        Spacer(1, 14),
        Paragraph("2. 효율 및 계산 조건", title_style),
        Spacer(1, 6),
        report_table([
            ["항목", "값"],
            ["적용 정격 모터 효율", f"{result['applied_eta_pct']:.2f}%"],
            ["인버터 모터 효율", f"{result['eta_motor_inv_pct']:.2f}%"],
            ["인버터 자체 효율", f"{result['eta_inv_self_pct']:.2f}%"],
            ["인버터 종합 효율", f"{result['eta_total_pct']:.2f}%"],
            ["밸브 제어 모터 효율", f"{result['eta_motor_valve_pct']:.2f}%"],
        ], [230, 260]),
        Spacer(1, 14),
        Paragraph("3. 유량별 소비전력", title_style),
        Spacer(1, 6),
    ]

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=180, bbox_inches="tight")
    image_buffer.seek(0)
    items.append(RLImage(image_buffer, width=480, height=215))
    items.extend([
        Spacer(1, 12),
        Paragraph("4. 면책 조항 및 산출 근거 고지", title_style),
        Spacer(1, 5),
        Paragraph(
            "<b>[면책 조항]</b> 본 시뮬레이션 결과는 참조 효율표 및 표준 부하 손실 곡선에 기반한 추정치이며, "
            "실제 현장 운전 조건, 부하 특성, 모터 명판 및 제조사 시험성적서 효율에 따라 오차가 발생할 수 있습니다. "
            "중요 입찰 및 기술 제안 시에는 반드시 제조사 공식 스펙을 재확인하시기 바랍니다.<br/><br/>"
            "엑셀 기준은 원본 XLS 모터 효율표(ita)와 밸브 축동력비(deguti)를 사용하며, "
            "인버터 운전 시 고정손은 n, 부상손은 n² 모델을 따릅니다.",
            note_style,
        ),
    ])
    document.build(items)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# 4. 사이드바 입력
# =========================================================
st.markdown('<div class="app-title">고압인버터 에너지 절감 시뮬레이션</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">사내 입찰·기술제안용 검증 시스템</div>', unsafe_allow_html=True)

st.sidebar.header("운전 조건 설정")
mode_option = st.sidebar.radio("모터 효율 기준", ("사내 엑셀 기준", "IE3 참조 효율 기준"))
calc_mode = "excel" if mode_option == "사내 엑셀 기준" else "ie3"
in_pm = st.sidebar.number_input("Pump Data Shaft Power (kW)", value=724.9, min_value=1.0, step=10.0, format="%.1f")

if in_pm > 3000:
    st.sidebar.warning(
        f"⚠️ 입력하신 {in_pm:.0f}kW는 참조 효율표 범위(3,000kW 이하)를 초과합니다."
    )

in_flow = st.sidebar.number_input("평균 운전 유량 또는 속도 (%)", value=75.5, min_value=N_MIN, max_value=N_MAX, step=0.1, format="%.1f")
in_hours = st.sidebar.number_input("연간 운전 시간 (h)", value=8400, min_value=0, step=100)
in_price = st.sidebar.number_input("전기요금 (원/kWh)", value=120.4, min_value=0.0, step=0.1, format="%.1f")

st.sidebar.divider()
st.sidebar.subheader("💰 투자 경제성 분석")
in_inv_cost = st.sidebar.number_input("인버터 도입 및 공사 비용 (원)", value=150000000.0, min_value=0.0, step=1000000.0, format="%.0f")

st.sidebar.divider()
st.sidebar.subheader("현장 모터 명판 효율")
selected_table = EXCEL_MOTOR_TABLE if calc_mode == "excel" else IE3_MOTOR_TABLE
default_eta = lookup_interp(in_pm, selected_table)
use_nameplate = st.sidebar.checkbox("명판 효율 직접 입력", value=False)
nameplate_eta = None
if use_nameplate:
    nameplate_eta = st.sidebar.number_input("명판 정격 효율 (%)", value=float(default_eta), min_value=50.0, max_value=99.9, step=0.1, format="%.2f")
else:
    st.sidebar.caption(f"현재 표 기준 정격 효율: {default_eta:.2f}%")

st.sidebar.divider()
st.sidebar.subheader("계산 내용 표시")
show_steps = st.sidebar.checkbox("계산 과정 보기", value=False)
show_losses = st.sidebar.checkbox("손실 모델 보기", value=False)
show_table = st.sidebar.checkbox("적용 효율표 보기", value=False)
show_compare = st.sidebar.checkbox("엑셀 기준과 IE3 비교", value=False)
show_ie3_basis = st.sidebar.checkbox("IE3 적용 기준 및 근거 보기", value=False)


# =========================================================
# 5. 메인 화면 및 비즈니스 지표 연산
# =========================================================
try:
    res = calculate_all(in_pm, in_flow, calc_mode, nameplate_eta)
except ValueError as error:
    st.error(str(error))
    st.stop()

saving_won = res["saving_kw"] * in_hours * in_price
payback_years = (in_inv_cost / saving_won) if saving_won > 0 else None

fig = generate_figure(in_pm, in_flow, calc_mode, nameplate_eta, res)
pdf_data = generate_pdf_report(res, mode_option, in_pm, in_flow, in_hours, in_price, in_inv_cost, payback_years, fig)

st.sidebar.divider()
st.sidebar.download_button(
    "PDF 보고서 다운로드",
    data=pdf_data,
    file_name=f"고압인버터_절감보고서_{in_pm:.0f}kW_{in_flow:.1f}pct.pdf",
    mime="application/pdf",
    use_container_width=True,
)

st.markdown("### 핵심 절감 및 투자 회수 요약")
payback_display = f"{payback_years:.2f} 년" if payback_years is not None else "산출 불가"

st.markdown(
    f"""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-label">Valve 제어 소비전력</div>
            <div class="summary-value">{res['P_valve_input']:,.1f} kW</div>
            <div class="summary-note">모터 효율 {res['eta_motor_valve_pct']:.1f}%</div>
        </div>
        <div class="summary-card summary-card-blue">
            <div class="summary-label">인버터 제어 소비전력</div>
            <div class="summary-value">{res['P_inv_input']:,.1f} kW</div>
            <div class="summary-note">종합 효율 {res['eta_total_pct']:.1f}%</div>
        </div>
        <div class="summary-card summary-card-green">
            <div class="summary-label">연간 절감 금액</div>
            <div class="summary-value summary-value-green">{saving_won / 1e8:.2f} 억원</div>
            <div class="summary-note">절감 전력 {res['saving_kw']:,.1f} kW ({res['saving_pct']:.1f}%)</div>
        </div>
        <div class="summary-card summary-card-green">
            <div class="summary-label">예상 투자 회수기간</div>
            <div class="summary-value summary-value-green">{payback_display}</div>
            <div class="summary-note">도입비용 {in_inv_cost/1e8:,.2f}억원 기준</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"현재 모터 효율 적용 기준: {res['efficiency_source']} ({res['applied_eta_pct']:.2f}%)")

st.markdown("### 세부 효율 비교")
detail_df = pd.DataFrame({
    "항목": ["소요 축동력 (kW)", "모터 효율 (%)", "인버터 자체 효율 (%)", "시스템 종합 효율 (%)", "최종 소비전력 (kW)"],
    "밸브 제어": [fmt(res["Pm_valve"]), fmt(res["eta_motor_valve_pct"]), "-", fmt(res["eta_motor_valve_pct"]), fmt(res["P_valve_input"])],
    "인버터 제어": [fmt(res["Pm_inv"]), fmt(res["eta_motor_inv_pct"]), fmt(res["eta_inv_self_pct"]), fmt(res["eta_total_pct"]), fmt(res["P_inv_input"])],
})
st.dataframe(detail_df, hide_index=True, use_container_width=True)

st.markdown("### 유량 변화에 따른 소비전력")
st.pyplot(fig)

if show_steps:
    st.markdown("### 계산 과정")
    step_df = pd.DataFrame([
        ["운전 속도비 n", f"{res['speed_ratio']:.4f}"],
        ["인버터 축동력", f"Pm × n³ = {res['Pm_inv']:,.3f} kW"],
        ["밸브 축동력 비율", f"{res['valve_ratio_pct']:.3f}%"],
        ["인버터 입력전력", f"{res['P_inv_input']:,.3f} kW"],
        ["밸브 입력전력", f"{res['P_valve_input']:,.3f} kW"],
    ], columns=["계산 항목", "값"])
    st.dataframe(step_df, hide_index=True, use_container_width=True)

if show_losses:
    st.markdown("### 손실 모델 상세")
    loss_df = pd.DataFrame([
        ["적용 정격 모터 효율", f"{res['applied_eta_pct']:.3f}%"],
        ["정격 총손실", f"{res['total_loss_kw']:.3f} kW"],
        ["고정손 비율", f"{res['fixed_loss_fraction'] * 100:.3f}%"],
        ["부하손 비율", f"{res['load_loss_fraction'] * 100:.3f}%"],
        ["인버터 운전 손실", f"{res['loss_inv_kw']:.3f} kW"],
        ["밸브 제어 손실", f"{res['loss_valve_kw']:.3f} kW"],
    ], columns=["항목", "값"])
    st.dataframe(loss_df, hide_index=True, use_container_width=True)

if show_table:
    st.markdown("### 적용 효율표")
    table_name = "사내 엑셀 모터 효율표" if calc_mode == "excel" else "IE3 참조 모터 효율표"
    st.caption(table_name)
    st.dataframe(pd.DataFrame(selected_table, columns=["용량 (kW)", "정격 효율 (%)"]), hide_index=True, use_container_width=True)

if show_compare:
    st.markdown("### 엑셀 기준과 IE3 기준 비교")
    excel_result = calculate_all(in_pm, in_flow, "excel", nameplate_eta)
    ie3_result = calculate_all(in_pm, in_flow, "ie3", nameplate_eta)
    compare_df = pd.DataFrame([
        ["적용 모터 효율 (%)", excel_result["applied_eta_pct"], ie3_result["applied_eta_pct"]],
        ["인버터 소비전력 (kW)", excel_result["P_inv_input"], ie3_result["P_inv_input"]],
        ["밸브 소비전력 (kW)", excel_result["P_valve_input"], ie3_result["P_valve_input"]],
        ["절감 전력 (kW)", excel_result["saving_kw"], ie3_result["saving_kw"]],
        ["절감률 (%)", excel_result["saving_pct"], ie3_result["saving_pct"]],
    ], columns=["항목", "사내 엑셀 기준", "IE3 참조 기준"])
    st.dataframe(compare_df.style.format({"사내 엑셀 기준": "{:,.2f}", "IE3 참조 기준": "{:,.2f}"}), hide_index=True, use_container_width=True)

if show_ie3_basis:
    st.markdown("### IE3 적용 기준 및 근거")
    st.markdown(
        """
        - **IEC 60034-30-1**은 선로 전원으로 운전하는 AC 모터의 IE 효율등급을 규정합니다.
        - 일반 적용 범위는 **0.12~1,000 kW, 1 kV 이하**입니다.
        - 인버터 운전 시 시스템 손실은 모터 효율등급에 직접 포함되지 않으므로, 중요 제안 시에는 현장 명판 효율을 우선 적용해야 합니다.
        
        공식 규격 문서: [IEC 60034-30-1 Preview Link](https://webstore.iec.ch/en/iec_catalog/product/preview/?id=L3B1Yi9wZGYvcHJldmlldy9pbmZvX2llYzYwMDM0LTMwLTF7ZWd0LjB9Yi5wZGY%3D)
        """
    )
