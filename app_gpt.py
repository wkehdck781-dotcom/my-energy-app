import io
import os

import matplotlib
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
# 1. 한글 폰트 동적 감지 및 등록
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "NanumGothic-Regular.ttf",
)

PDF_FONT = "Helvetica"

# Matplotlib에서 직접 사용할 FontProperties
KOREAN_FONT = None


def setup_fonts():

    global KOREAN_FONT

    matplotlib_font_name = "DejaVu Sans"
    pdf_font_name = "Helvetica"

    # -----------------------------------------------------
    # 폰트 파일 존재 확인
    # -----------------------------------------------------
    if not os.path.isfile(FONT_PATH):

        print(
            "[WARNING] 한글 폰트 파일을 찾을 수 없습니다."
        )

        print(
            f"[WARNING] 확인 경로: {FONT_PATH}"
        )

        # 기본 폰트 사용
        matplotlib.rcParams["font.family"] = [
            "DejaVu Sans"
        ]

        matplotlib.rcParams[
            "axes.unicode_minus"
        ] = False

        return pdf_font_name

    # -----------------------------------------------------
    # Matplotlib 폰트 등록
    # -----------------------------------------------------
    try:

        fm.fontManager.addfont(FONT_PATH)

        KOREAN_FONT = fm.FontProperties(
            fname=FONT_PATH
        )

        matplotlib_font_name = (
            KOREAN_FONT.get_name()
        )

        matplotlib.rcParams["font.family"] = [
            matplotlib_font_name
        ]

        matplotlib.rcParams[
            "axes.unicode_minus"
        ] = False

        print(
            "[OK] Matplotlib 한글 폰트 등록:"
            f" {matplotlib_font_name}"
        )

    except Exception as e:

        KOREAN_FONT = None

        print(
            "[WARNING] Matplotlib 폰트 등록 실패:"
            f" {repr(e)}"
        )

        matplotlib.rcParams["font.family"] = [
            "DejaVu Sans"
        ]

        matplotlib.rcParams[
            "axes.unicode_minus"
        ] = False

    # -----------------------------------------------------
    # ReportLab PDF 폰트 등록
    # -----------------------------------------------------
    try:

        registered_fonts = (
            pdfmetrics.getRegisteredFontNames()
        )

        if "NanumGothicCustom" not in registered_fonts:

            pdfmetrics.registerFont(
                TTFont(
                    "NanumGothicCustom",
                    FONT_PATH,
                )
            )

        pdf_font_name = "NanumGothicCustom"

        print(
            "[OK] ReportLab 한글 폰트 등록 완료"
        )

    except Exception as e:

        print(
            "[WARNING] ReportLab 폰트 등록 실패:"
            f" {repr(e)}"
        )

        pdf_font_name = "Helvetica"

    return pdf_font_name


PDF_FONT = setup_fonts()


# =========================================================
# 2. 색상 설정
# =========================================================

COLOR_INK = "#172033"
COLOR_MUTED = "#6B7280"
COLOR_LINE = "#D8DEE9"
COLOR_ACCENT = "#2563EB"
COLOR_GREEN = "#15803D"
COLOR_GREEN_BG = "#F0FDF4"
COLOR_BLUE_BG = "#EFF6FF"


# =========================================================
# 3. Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="고압인버터 에너지 절감 시뮬레이터",
    layout="wide",
)


# =========================================================
# 4. 화면 CSS
# =========================================================

st.markdown(
    f"""
    <style>

    .app-title {{
        font-size: 26px;
        font-weight: 750;
        color: {COLOR_INK};
        margin-bottom: 4px;
    }}

    .app-subtitle {{
        color: {COLOR_MUTED};
        font-size: 14px;
        margin-bottom: 24px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. 참조 데이터 테이블
# =========================================================

EXCEL_MOTOR_TABLE = [
    (0, 91.0),
    (55, 91.0),
    (90, 92.0),
    (132, 93.0),
    (160, 90.0),
    (400, 94.0),
    (630, 94.5),
    (800, 94.5),
    (1250, 96.0),
    (1800, 96.5),
    (3000, 97.0),
]


IE3_MOTOR_TABLE = [
    (0, 85.5),
    (55, 95.0),
    (90, 95.4),
    (132, 95.8),
    (160, 95.8),
    (375, 96.2),
    (630, 96.5),
    (800, 96.7),
    (1250, 97.1),
    (1800, 97.3),
    (3000, 97.5),
]


VALVE_RATIO_TABLE = [
    (53.5, 79.7571428571428),
    (60.2, 82.5589507464769),
    (66.8, 84.8751220873447),
    (75.5, 89.3400306962467),
    (80.2, 91.8515417887540),
    (83.4, 93.5956545730446),
    (93.6, 97.6698758197293),
    (100.0, 100.0),
]


N_MIN = 53.5
N_MAX = 100.0


# =========================================================
# 6. 계산 함수
# =========================================================

def lookup_interp(x, table):

    return float(
        np.interp(
            x,
            [row[0] for row in table],
            [row[1] for row in table],
        )
    )


def loss_profile_from_excel(
    pump_rated_shaft_kw
):

    excel_eta_pct = lookup_interp(
        pump_rated_shaft_kw,
        EXCEL_MOTOR_TABLE,
    )

    eta = excel_eta_pct / 100.0

    total_loss_kw = (
        pump_rated_shaft_kw / eta
        - pump_rated_shaft_kw
    )

    fixed_loss_fraction = (
        pump_rated_shaft_kw
        * 0.0333
        / total_loss_kw
    )

    load_loss_fraction = (
        1.0
        - fixed_loss_fraction
    )

    return (
        fixed_loss_fraction,
        load_loss_fraction,
        excel_eta_pct,
    )


def inverter_efficiency_pct(
    speed_ratio
):

    if speed_ratio <= 0:
        return 0.0

    if speed_ratio <= 0.8:
        return 95.9

    return (
        98.1
        - (
            0.5
            + 0.6 * speed_ratio
        )
        / speed_ratio**3
    )


@st.cache_data
def calculate_all(
    pump_rated_shaft_kw,
    flow_pct,
    mode,
    nameplate_efficiency_pct=None,
):

    if pump_rated_shaft_kw <= 0:

        raise ValueError(
            "Pump Data Shaft Power는 0보다 커야 합니다."
        )

    if not N_MIN <= flow_pct <= N_MAX:

        raise ValueError(
            f"유량/속도는 "
            f"{N_MIN}~{N_MAX}% 범위여야 합니다."
        )

    selected_table = (
        EXCEL_MOTOR_TABLE
        if mode == "excel"
        else IE3_MOTOR_TABLE
    )

    reference_eta_pct = lookup_interp(
        pump_rated_shaft_kw,
        selected_table,
    )

    if nameplate_efficiency_pct is None:

        applied_eta_pct = (
            reference_eta_pct
        )

        efficiency_source = (
            "사내 엑셀 효율표"
            if mode == "excel"
            else "IE3 참조 효율표"
        )

    else:

        applied_eta_pct = (
            nameplate_efficiency_pct
        )

        efficiency_source = (
            "현장 모터 명판 입력값"
        )

    n = flow_pct / 100.0

    (
        fixed_loss_fraction,
        load_loss_fraction,
        excel_eta_pct,
    ) = loss_profile_from_excel(
        pump_rated_shaft_kw
    )

    total_loss_kw = (
        pump_rated_shaft_kw
        / (applied_eta_pct / 100.0)
        - pump_rated_shaft_kw
    )

    # -----------------------------------------------------
    # 인버터 운전
    # -----------------------------------------------------

    pm_inv = (
        pump_rated_shaft_kw
        * n**3
    )

    loss_inv_kw = (
        fixed_loss_fraction * n
        + load_loss_fraction * n**2
    ) * total_loss_kw

    eta_motor_inv = (
        pm_inv
        / (pm_inv + loss_inv_kw)
    )

    eta_inv_pct = (
        inverter_efficiency_pct(n)
    )

    eta_total = (
        eta_motor_inv
        * eta_inv_pct
        / 100.0
    )

    p_inv_input = (
        pm_inv / eta_total
    )

    # -----------------------------------------------------
    # 밸브 제어
    # -----------------------------------------------------

    valve_ratio = (
        lookup_interp(
            flow_pct,
            VALVE_RATIO_TABLE,
        )
        / 100.0
    )

    pm_valve = (
        pump_rated_shaft_kw
        * valve_ratio
    )

    loss_valve_kw = (
        fixed_loss_fraction
        + load_loss_fraction
        * valve_ratio**2
    ) * total_loss_kw

    eta_motor_valve = (
        pm_valve
        / (pm_valve + loss_valve_kw)
    )

    p_valve_input = (
        pm_valve
        / eta_motor_valve
    )

    # -----------------------------------------------------
    # 절감량
    # -----------------------------------------------------

    saving_kw = (
        p_valve_input
        - p_inv_input
    )

    return {

        "mode": mode,

        "efficiency_source":
            efficiency_source,

        "reference_eta_pct":
            reference_eta_pct,

        "applied_eta_pct":
            applied_eta_pct,

        "excel_eta_pct":
            excel_eta_pct,

        "speed_ratio":
            n,

        "total_loss_kw":
            total_loss_kw,

        "fixed_loss_fraction":
            fixed_loss_fraction,

        "load_loss_fraction":
            load_loss_fraction,

        "Pm_inv":
            pm_inv,

        "loss_inv_kw":
            loss_inv_kw,

        "eta_motor_inv_pct":
            eta_motor_inv * 100.0,

        "eta_inv_self_pct":
            eta_inv_pct,

        "eta_total_pct":
            eta_total * 100.0,

        "P_inv_input":
            p_inv_input,

        "valve_ratio_pct":
            valve_ratio * 100.0,

        "Pm_valve":
            pm_valve,

        "loss_valve_kw":
            loss_valve_kw,

        "eta_motor_valve_pct":
            eta_motor_valve * 100.0,

        "P_valve_input":
            p_valve_input,

        "saving_kw":
            saving_kw,

        "saving_pct":
            saving_kw
            / p_valve_input
            * 100.0,
    }


def fmt(
    value,
    digits=1
):

    if value is None:
        return "-"

    return f"{value:,.{digits}f}"


# =========================================================
# 7. 그래프 생성
# =========================================================

def generate_figure(
    pump_kw,
    flow_pct,
    mode,
    nameplate_efficiency_pct,
    result,
):

    flows = np.linspace(
        N_MIN,
        N_MAX,
        100,
    )

    curve_inv = []
    curve_valve = []

    for flow in flows:

        curve_result = calculate_all(
            pump_kw,
            float(flow),
            mode,
            nameplate_efficiency_pct,
        )

        curve_inv.append(
            curve_result["P_inv_input"]
        )

        curve_valve.append(
            curve_result["P_valve_input"]
        )

    fig, ax = plt.subplots(
        figsize=(10, 4.5),
        dpi=200,
    )

    # -----------------------------------------------------
    # 선 그래프
    # -----------------------------------------------------

    ax.plot(
        flows,
        curve_valve,
        color="#94A3B8",
        label="밸브 제어 (기존)",
        linewidth=2.5,
    )

    ax.plot(
        flows,
        curve_inv,
        color=COLOR_ACCENT,
        label="인버터 제어 (적용)",
        linewidth=3.0,
    )

    ax.fill_between(
        flows,
        curve_inv,
        curve_valve,
        color="#38BDF8",
        alpha=0.2,
        label="에너지 절감 구간",
    )

    # -----------------------------------------------------
    # 현재 운전점
    # -----------------------------------------------------

    ax.scatter(
        [flow_pct],
        [result["P_valve_input"]],
        color="#64748B",
        s=60,
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )

    ax.scatter(
        [flow_pct],
        [result["P_inv_input"]],
        color=COLOR_ACCENT,
        s=70,
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )

    ax.axvline(
        x=flow_pct,
        color="#CBD5E1",
        linestyle=":",
        linewidth=1.5,
    )

    # -----------------------------------------------------
    # 한글 폰트 직접 지정
    # -----------------------------------------------------

    if KOREAN_FONT is not None:

        ax.set_xlabel(
            "유량 / 속도 비율 (%)",
            fontproperties=KOREAN_FONT,
            fontsize=11,
            fontweight="bold",
            labelpad=8,
        )

        ax.set_ylabel(
            "소비전력 (kW)",
            fontproperties=KOREAN_FONT,
            fontsize=11,
            fontweight="bold",
            labelpad=8,
        )

        ax.set_title(
            "유량 변화에 따른 시스템 소비전력 비교 곡선",
            fontproperties=KOREAN_FONT,
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        legend = ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
            fontsize=9,
            prop=KOREAN_FONT,
        )

    else:

        ax.set_xlabel(
            "유량 / 속도 비율 (%)",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
        )

        ax.set_ylabel(
            "소비전력 (kW)",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
        )

        ax.set_title(
            "유량 변화에 따른 시스템 소비전력 비교 곡선",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        legend = ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
            fontsize=9,
        )

    # -----------------------------------------------------
    # 축/그리드
    # -----------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
        color="#E2E8F0",
    )

    for spine in (
        "top",
        "right",
    ):

        ax.spines[
            spine
        ].set_visible(False)

    for spine in (
        "left",
        "bottom",
    ):

        ax.spines[
            spine
        ].set_color("#94A3B8")

    plt.tight_layout()

    return fig


# =========================================================
# 8. PDF 보고서 생성
# =========================================================

def generate_pdf_report(
    result,
    mode_name,
    pump_kw,
    flow_pct,
    hours,
    price,
    inv_cost,
    payback_years,
    figure,
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PdfTitle",
        parent=styles["Heading1"],
        fontName=PDF_FONT,
        fontSize=15,
        leading=19,
        textColor=colors.HexColor(
            COLOR_INK
        ),
    )

    normal_style = ParagraphStyle(
        "PdfNormal",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor(
            COLOR_INK
        ),
    )

    header_style = ParagraphStyle(
        "PdfHeader",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.white,
    )

    cell_style = ParagraphStyle(
        "PdfCell",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor(
            COLOR_INK
        ),
    )

    note_style = ParagraphStyle(
        "PdfNote",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=7.2,
        leading=10,
        textColor=colors.HexColor(
            COLOR_MUTED
        ),
    )

    # -----------------------------------------------------
    # PDF Table 함수
    # -----------------------------------------------------

    def report_table(
        rows,
        widths,
    ):

        prepared = []

        for row_index, row in enumerate(rows):

            style = (
                header_style
                if row_index == 0
                else cell_style
            )

            prepared.append(
                [
                    Paragraph(
                        str(value),
                        style,
                    )
                    for value in row
                ]
            )

        table = Table(
            prepared,
            colWidths=widths,
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            COLOR_INK
                        ),
                    ),

                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.HexColor(
                            "#F8FAFC"
                        ),
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            COLOR_LINE
                        ),
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        return table

    # -----------------------------------------------------
    # 연간 절감금액
    # -----------------------------------------------------

    saving_won = (
        result["saving_kw"]
        * hours
        * price
    )

    payback_str = (
        f"{payback_years:.2f} 년"
        if payback_years is not None
        else "산출 불가 (절감액 없음)"
    )

    # -----------------------------------------------------
    # PDF 구성
    # -----------------------------------------------------

    items = [

        Paragraph(
            "고압인버터 에너지 절감 시뮬레이션 보고서",
            title_style,
        ),

        Spacer(1, 5),

        Paragraph(
            f"적용 기준: {mode_name} / "
            f"모터 효율: "
            f"{result['efficiency_source']} "
            f"({result['applied_eta_pct']:.2f}%)"
            f"<br/>"
            f"Pump Shaft Power: "
            f"{pump_kw:.1f} kW / "
            f"운전 유량: "
            f"{flow_pct:.1f}% / "
            f"연간 운전시간: "
            f"{hours:,.0f} h / "
            f"전력단가: "
            f"{price:,.1f} 원/kWh",
            normal_style,
        ),

        Spacer(1, 14),

        Paragraph(
            "1. 핵심 절감 및 투자 회수 요약",
            title_style,
        ),

        Spacer(1, 6),

        report_table(
            [
                [
                    "항목",
                    "Valve 제어",
                    "인버터 제어",
                    "절감 효과",
                ],

                [
                    "소비전력",
                    f"{result['P_valve_input']:,.1f} kW",
                    f"{result['P_inv_input']:,.1f} kW",
                    f"{result['saving_kw']:,.1f} kW",
                ],

                [
                    "절감률",
                    "-",
                    "-",
                    f"{result['saving_pct']:.1f}%",
                ],

                [
                    "연간 절감 금액",
                    "-",
                    "-",
                    f"{saving_won / 1e8:.2f} 억원/년",
                ],

                [
                    "예상 투자 회수기간",
                    "-",
                    "-",
                    payback_str,
                ],
            ],
            [
                115,
                115,
                115,
                145,
            ],
        ),

        Spacer(1, 14),

        Paragraph(
            "2. 효율 및 계산 조건",
            title_style,
        ),

        Spacer(1, 6),

        report_table(
            [
                [
                    "항목",
                    "값",
                ],

                [
                    "적용 정격 모터 효율",
                    f"{result['applied_eta_pct']:.2f}%",
                ],

                [
                    "인버터 모터 효율",
                    f"{result['eta_motor_inv_pct']:.2f}%",
                ],

                [
                    "인버터 자체 효율",
                    f"{result['eta_inv_self_pct']:.2f}%",
                ],

                [
                    "인버터 종합 효율",
                    f"{result['eta_total_pct']:.2f}%",
                ],

                [
                    "밸브 제어 모터 효율",
                    f"{result['eta_motor_valve_pct']:.2f}%",
                ],
            ],
            [
                230,
                260,
            ],
        ),

        Spacer(1, 14),

        Paragraph(
            "3. 유량별 소비전력",
            title_style,
        ),

        Spacer(1, 6),
    ]

    # -----------------------------------------------------
    # 그래프 PDF 삽입
    # -----------------------------------------------------

    image_buffer = io.BytesIO()

    figure.savefig(
        image_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    image_buffer.seek(0)

    items.append(
        RLImage(
            image_buffer,
            width=480,
            height=215,
        )
    )

    # -----------------------------------------------------
    # 면책 조항
    # -----------------------------------------------------

    items.extend(
        [

            Spacer(1, 12),

            Paragraph(
                "4. 면책 조항 및 산출 근거 고지",
                title_style,
            ),

            Spacer(1, 5),

            Paragraph(
                "<b>[면책 조항]</b> "
                "본 시뮬레이션 결과는 참조 효율표 및 "
                "표준 부하 손실 곡선에 기반한 추정치이며, "
                "실제 현장 운전 조건, 부하 특성, 모터 명판 및 "
                "제조사 시험성적서 효율에 따라 오차가 발생할 수 있습니다. "
                "중요 입찰 및 기술 제안 시에는 반드시 제조사 공식 스펙을 "
                "재확인하시기 바랍니다."
                "<br/><br/>"
                "엑셀 기준은 원본 XLS 모터 효율표(ita)와 "
                "밸브 축동력비(deguti)를 사용하며, "
                "인버터 운전 시 고정손은 n, 부상손은 n² 모델을 따릅니다.",
                note_style,
            ),
        ]
    )

    document.build(items)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# 9. 메인 화면
# =========================================================

st.markdown(
    '<div class="app-title">'
    '고압인버터 에너지 절감 시뮬레이션'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    '사내 입찰·기술제안용 검증 시스템'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 10. 사이드바 입력
# =========================================================

st.sidebar.header(
    "운전 조건 설정"
)


mode_option = st.sidebar.radio(
    "모터 효율 기준",
    (
        "사내 엑셀 기준",
        "IE3 참조 효율 기준",
    ),
)


calc_mode = (
    "excel"
    if mode_option == "사내 엑셀 기준"
    else "ie3"
)


in_pm = st.sidebar.number_input(
    "Pump Data Shaft Power (kW)",
    value=724.9,
    min_value=1.0,
    step=10.0,
    format="%.1f",
)


if in_pm > 3000:

    st.sidebar.warning(
        f"⚠️ 입력하신 {in_pm:.0f}kW는 "
        "참조 효율표 범위(3,000kW 이하)를 초과합니다."
    )


in_flow = st.sidebar.number_input(
    "평균 운전 유량 또는 속도 (%)",
    value=75.5,
    min_value=N_MIN,
    max_value=N_MAX,
    step=0.1,
    format="%.1f",
)


in_hours = st.sidebar.number_input(
    "연간 운전 시간 (h)",
    value=8400,
    min_value=0,
    step=100,
)


in_price = st.sidebar.number_input(
    "전기요금 (원/kWh)",
    value=120.4,
    min_value=0.0,
    step=0.1,
    format="%.1f",
)


st.sidebar.divider()


# =========================================================
# 11. 투자 경제성 분석
# =========================================================

st.sidebar.subheader(
    "💰 투자 경제성 분석"
)


in_inv_cost = st.sidebar.number_input(
    "인버터 도입 및 공사 비용 (원)",
    value=150000000.0,
    min_value=0.0,
    step=1000000.0,
    format="%.0f",
)


st.sidebar.divider()


# =========================================================
# 12. 현장 모터 명판 효율
# =========================================================

st.sidebar.subheader(
    "현장 모터 명판 효율"
)


selected_table = (
    EXCEL_MOTOR_TABLE
    if calc_mode == "excel"
    else IE3_MOTOR_TABLE
)


default_eta = lookup_interp(
    in_pm,
    selected_table,
)


use_nameplate = st.sidebar.checkbox(
    "명판 효율 직접 입력",
    value=False,
)


nameplate_eta = None


if use_nameplate:

    nameplate_eta = st.sidebar.number_input(
        "명판 정격 효율 (%)",
        value=float(default_eta),
        min_value=50.0,
        max_value=99.9,
        step=0.1,
        format="%.2f",
    )

else:

    st.sidebar.caption(
        f"현재 표 기준 정격 효율: "
        f"{default_eta:.2f}%"
    )


# =========================================================
# 13. 계산 내용 표시 옵션
# =========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "계산 내용 표시"
)


show_steps = st.sidebar.checkbox(
    "계산 과정 보기",
    value=False,
)


show_losses = st.sidebar.checkbox(
    "손실 모델 보기",
    value=False,
)


show_table = st.sidebar.checkbox(
    "적용 효율표 보기",
    value=False,
)


show_compare = st.sidebar.checkbox(
    "엑셀 기준과 IE3 비교",
    value=False,
)


show_ie3_basis = st.sidebar.checkbox(
    "IE3 적용 기준 및 근거 보기",
    value=False,
)


# =========================================================
# 14. 메인 계산
# =========================================================

try:

    res = calculate_all(
        in_pm,
        in_flow,
        calc_mode,
        nameplate_eta,
    )

except ValueError as error:

    st.error(
        str(error)
    )

    st.stop()


# =========================================================
# 15. 경제성 계산
# =========================================================

saving_won = (
    res["saving_kw"]
    * in_hours
    * in_price
)


payback_years = (
    in_inv_cost / saving_won
    if saving_won > 0
    else None
)


# =========================================================
# 16. 그래프 및 PDF 생성
# =========================================================

fig = generate_figure(
    in_pm,
    in_flow,
    calc_mode,
    nameplate_eta,
    res,
)


pdf_data = generate_pdf_report(
    res,
    mode_option,
    in_pm,
    in_flow,
    in_hours,
    in_price,
    in_inv_cost,
    payback_years,
    fig,
)


# =========================================================
# 17. PDF 다운로드
# =========================================================

st.sidebar.divider()


st.sidebar.download_button(
    "PDF 보고서 다운로드",
    data=pdf_data,
    file_name=(
        f"고압인버터_절감보고서_"
        f"{in_pm:.0f}kW_"
        f"{in_flow:.1f}pct.pdf"
    ),
    mime="application/pdf",
    use_container_width=True,
)


# =========================================================
# 18. 핵심 절감 및 투자 회수 요약
# =========================================================

st.markdown(
    "### 핵심 절감 및 투자 회수 요약"
)


payback_display = (
    f"{payback_years:.2f} 년"
    if payback_years is not None
    else "산출 불가"
)


# ---------------------------------------------------------
# 기존 HTML 카드 제거
# st.markdown + <div> 방식 대신 Streamlit 기본 UI 사용
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="Valve 제어 소비전력",
        value=(
            f"{res['P_valve_input']:,.1f} kW"
        ),
        delta=(
            f"모터 효율 "
            f"{res['eta_motor_valve_pct']:.1f}%"
        ),
        delta_color="off",
    )


with col2:

    st.metric(
        label="인버터 제어 소비전력",
        value=(
            f"{res['P_inv_input']:,.1f} kW"
        ),
        delta=(
            f"종합 효율 "
            f"{res['eta_total_pct']:.1f}%"
        ),
        delta_color="off",
    )


with col3:

    st.metric(
        label="연간 절감 금액",
        value=(
            f"{saving_won / 1e8:.2f} 억원"
        ),
        delta=(
            f"절감 전력 "
            f"{res['saving_kw']:,.1f} kW "
            f"({res['saving_pct']:.1f}%)"
        ),
        delta_color="normal",
    )


with col4:

    st.metric(
        label="예상 투자 회수기간",
        value=payback_display,
        delta=(
            f"도입비용 "
            f"{in_inv_cost / 1e8:.2f}억원 기준"
        ),
        delta_color="off",
    )


st.caption(
    f"현재 모터 효율 적용 기준: "
    f"{res['efficiency_source']} "
    f"({res['applied_eta_pct']:.2f}%)"
)


# =========================================================
# 19. 세부 효율 비교
# =========================================================

st.markdown(
    "### 세부 효율 비교"
)


detail_df = pd.DataFrame(
    {

        "항목": [
            "소요 축동력 (kW)",
            "모터 효율 (%)",
            "인버터 자체 효율 (%)",
            "시스템 종합 효율 (%)",
            "최종 소비전력 (kW)",
        ],

        "밸브 제어": [
            fmt(
                res["Pm_valve"]
            ),

            fmt(
                res["eta_motor_valve_pct"]
            ),

            "-",

            fmt(
                res["eta_motor_valve_pct"]
            ),

            fmt(
                res["P_valve_input"]
            ),
        ],

        "인버터 제어": [
            fmt(
                res["Pm_inv"]
            ),

            fmt(
                res["eta_motor_inv_pct"]
            ),

            fmt(
                res["eta_inv_self_pct"]
            ),

            fmt(
                res["eta_total_pct"]
            ),

            fmt(
                res["P_inv_input"]
            ),
        ],
    }
)


st.dataframe(
    detail_df,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# 20. 유량 변화에 따른 소비전력 그래프
# =========================================================

st.markdown(
    "### 유량 변화에 따른 소비전력"
)


st.pyplot(
    fig,
    use_container_width=True,
)


# =========================================================
# 21. 계산 과정
# =========================================================

if show_steps:

    st.markdown(
        "### 계산 과정"
    )

    step_df = pd.DataFrame(
        [

            [
                "운전 속도비 n",
                f"{res['speed_ratio']:.4f}",
            ],

            [
                "인버터 축동력",
                f"Pm × n³ = "
                f"{res['Pm_inv']:,.3f} kW",
            ],

            [
                "밸브 축동력 비율",
                f"{res['valve_ratio_pct']:.3f}%",
            ],

            [
                "인버터 입력전력",
                f"{res['P_inv_input']:,.3f} kW",
            ],

            [
                "밸브 입력전력",
                f"{res['P_valve_input']:,.3f} kW",
            ],
        ],

        columns=[
            "계산 항목",
            "값",
        ],
    )

    st.dataframe(
        step_df,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 22. 손실 모델 상세
# =========================================================

if show_losses:

    st.markdown(
        "### 손실 모델 상세"
    )

    loss_df = pd.DataFrame(
        [

            [
                "적용 정격 모터 효율",
                f"{res['applied_eta_pct']:.3f}%",
            ],

            [
                "정격 총손실",
                f"{res['total_loss_kw']:.3f} kW",
            ],

            [
                "고정손 비율",
                f"{res['fixed_loss_fraction'] * 100:.3f}%",
            ],

            [
                "부하손 비율",
                f"{res['load_loss_fraction'] * 100:.3f}%",
            ],

            [
                "인버터 운전 손실",
                f"{res['loss_inv_kw']:.3f} kW",
            ],

            [
                "밸브 제어 손실",
                f"{res['loss_valve_kw']:.3f} kW",
            ],
        ],

        columns=[
            "항목",
            "값",
        ],
    )

    st.dataframe(
        loss_df,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 23. 적용 효율표
# =========================================================

if show_table:

    st.markdown(
        "### 적용 효율표"
    )

    table_name = (
        "사내 엑셀 모터 효율표"
        if calc_mode == "excel"
        else "IE3 참조 모터 효율표"
    )

    st.caption(
        table_name
    )

    st.dataframe(
        pd.DataFrame(
            selected_table,
            columns=[
                "용량 (kW)",
                "정격 효율 (%)",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 24. 엑셀 기준과 IE3 기준 비교
# =========================================================

if show_compare:

    st.markdown(
        "### 엑셀 기준과 IE3 기준 비교"
    )

    excel_result = calculate_all(
        in_pm,
        in_flow,
        "excel",
        nameplate_eta,
    )

    ie3_result = calculate_all(
        in_pm,
        in_flow,
        "ie3",
        nameplate_eta,
    )

    compare_df = pd.DataFrame(
        [

            [
                "적용 모터 효율 (%)",
                excel_result[
                    "applied_eta_pct"
                ],
                ie3_result[
                    "applied_eta_pct"
                ],
            ],

            [
                "인버터 소비전력 (kW)",
                excel_result[
                    "P_inv_input"
                ],
                ie3_result[
                    "P_inv_input"
                ],
            ],

            [
                "밸브 소비전력 (kW)",
                excel_result[
                    "P_valve_input"
                ],
                ie3_result[
                    "P_valve_input"
                ],
            ],

            [
                "절감 전력 (kW)",
                excel_result[
                    "saving_kw"
                ],
                ie3_result[
                    "saving_kw"
                ],
            ],

            [
                "절감률 (%)",
                excel_result[
                    "saving_pct"
                ],
                ie3_result[
                    "saving_pct"
                ],
            ],
        ],

        columns=[
            "항목",
            "사내 엑셀 기준",
            "IE3 참조 기준",
        ],
    )

    st.dataframe(
        compare_df.style.format(
            {
                "사내 엑셀 기준": "{:,.2f}",
                "IE3 참조 기준": "{:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# 25. IE3 적용 기준 및 근거
# =========================================================

if show_ie3_basis:

    st.markdown(
        "### IE3 적용 기준 및 근거"
    )

    st.markdown(
        """
        - **IEC 60034-30-1**은 선로 전원으로 운전하는 AC 모터의 IE 효율등급을 규정합니다.
        - 일반 적용 범위는 **0.12~1,000 kW, 1 kV 이하**입니다.
        - 인버터 운전 시 시스템 손실은 모터 효율등급에 직접 포함되지 않으므로, 중요 제안 시에는 현장 명판 효율을 우선 적용해야 합니다.

        공식 규격 문서:
        [IEC 60034-30-1 Preview Link](https://webstore.iec.ch/en/iec_catalog/product/preview/?id=L3B1Yi9wZGYvcHJldmlldy9pbmZvX2llYzYwMDM0LTMwLTF7ZWd0LjB9Yi5wZGY%3D)
        """
    )
