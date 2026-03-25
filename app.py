import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.simulation import Simulation, SimulationConfig
from src.metrics import (
    compute_exploitability,
    compute_pricing_error,
    compute_lp_metrics,
    compute_market_stability,
)

st.set_page_config(
    page_title="Options Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "sidebar_hidden": False,
    "show_params": True,
    "n_steps": 150,
    "twap_window": 2,
    "initial_price": 500,
    "attack_size": 50_000,
    "run_attack": True,
    "seed": 42,
    "run_btn": False,
    "has_run": False,
    "results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
*, body, html { font-family: 'Inter', sans-serif !important; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: #13111a !important; }
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }

[data-testid="stSidebar"] { background-color: #1a1727 !important; border-right: 1px solid #2d2640 !important; min-width: 280px !important; }
[data-testid="stSidebar"] * { color: #b8afd4 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p { color: #ede9f6 !important; }
[data-testid="stSidebar"]::before {
    content: "";
    display: block;
    height: 2px;
    background: linear-gradient(90deg, transparent, #fc72ff, #6c63ff, transparent);
    margin-bottom: 0.5rem;
}

h1, h2, h3, h4 { color: #ede9f6 !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
p, span, div, label { color: #b8afd4 !important; }

[data-testid="metric-container"] {
    background: #1a1727 !important;
    border: 1px solid #2d2640 !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    box-shadow: 0 0 0 1px #2d2640, 0 0 12px rgba(108,99,255,0.08) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 0 1px #fc72ff55, 0 0 24px rgba(252,114,255,0.12) !important;
}
[data-testid="stMetricValue"] { color: #ede9f6 !important; font-size: 1.6rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #6b5f8a !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stMetricDelta"] { color: #fc72ff !important; font-size: 0.75rem !important; }

[data-testid="stSlider"] [role="slider"] { background: #fc72ff !important; border: none !important; box-shadow: 0 0 0 3px rgba(252,114,255,0.2) !important; }
[data-baseweb="input"] input, input[type="number"] { background: #1a1727 !important; border: 1px solid #2d2640 !important; border-radius: 12px !important; color: #ede9f6 !important; padding: 0.5rem 0.75rem !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #fc72ff 0%, #b44fe8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.25rem !important;
    box-shadow: 0 0 16px rgba(252,114,255,0.3) !important;
    transition: box-shadow 0.2s ease, opacity 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 28px rgba(252,114,255,0.5) !important;
    opacity: 0.88 !important;
}

hr { border: none !important; border-top: 1px solid #2d2640 !important; margin: 1.5rem 0 !important; }

[data-testid="stDataFrame"] > div { background: #1a1727 !important; border: 1px solid #2d2640 !important; border-radius: 16px !important; overflow: hidden !important; }

[data-testid="stExpander"] {
    background: #1a1727 !important;
    border: 1px solid #2d2640 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 12px rgba(108,99,255,0.06) !important;
}
[data-testid="stExpander"] summary { color: #b8afd4 !important; }
[data-testid="stExpander"] summary p { color: #b8afd4 !important; font-size: 0.875rem !important; }
[data-testid="stExpander"] summary svg { display: none !important; }
[data-testid="stExpander"] details[open] summary::before { content: "↑ "; color: #6b5f8a; font-size: 0.85rem; font-family: sans-serif !important; }
[data-testid="stExpander"] summary::before { content: "↓ "; color: #6b5f8a; font-size: 0.85rem; font-family: sans-serif !important; }

[data-testid="stAlert"] { background: #1a1727 !important; border: 1px solid #2d2640 !important; border-radius: 16px !important; }
[data-testid="stAlert"] p { color: #b8afd4 !important; }

.opt-wrap { flex: 1; display: flex; flex-direction: column; box-shadow: 0 0 12px rgba(108,99,255,0.06) !important; }

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(108,99,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(108,99,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.title-gradient {
    background: linear-gradient(270deg, #fc72ff, #b44fe8, #6c63ff, #36d1b7);
    background-size: 400% 400%;
    animation: shimmer 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

@media (min-width: 768px) {
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
}
@media (max-width: 767px) {
    div[data-testid="column"]:last-child { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

if st.session_state.sidebar_hidden:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

components.html("""
<script>
(function() {
    const doc = window.parent.document;

    function getSidebarBtn() {
        return doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
            || doc.querySelector('[data-testid="collapsedControl"] button');
    }

    function hideOriginalBtn() {
        const style = doc.getElementById('hamburger-style') || doc.createElement('style');
        style.id = 'hamburger-style';
        style.textContent = `
            [data-testid="collapsedControl"] button,
            [data-testid="stSidebarCollapseButton"] button {
                color: transparent !important;
                font-size: 0 !important;
            }
            [data-testid="collapsedControl"] button svg,
            [data-testid="stSidebarCollapseButton"] button svg {
                visibility: hidden !important;
            }
            @media (max-width: 767px) {
                [data-testid="collapsedControl"] button,
                [data-testid="stSidebarCollapseButton"] button {
                    opacity: 1 !important;
                    position: fixed !important;
                    top: 0.75rem !important;
                    left: 0.75rem !important;
                    width: 2.2rem !important;
                    height: 2.2rem !important;
                    z-index: 999998 !important;
                    pointer-events: auto !important;
                    background: transparent !important;
                    border: none !important;
                }
            }
        `;
        if (!doc.getElementById('hamburger-style')) {
            doc.head.appendChild(style);
        }
    }

    function injectHamburger() {
        hideOriginalBtn();
        if (doc.getElementById('custom-hamburger')) return;

        const btn = doc.createElement('button');
        btn.id = 'custom-hamburger';
        btn.textContent = '☰';
        btn.style.cssText = `
            display: none;
            position: fixed;
            top: 0.75rem;
            left: 0.75rem;
            z-index: 999999;
            background: #1a1727;
            border: 1px solid #fc72ff;
            border-radius: 8px;
            width: 2.2rem;
            height: 2.2rem;
            font-size: 1.1rem;
            color: #fc72ff;
            cursor: pointer;
            align-items: center;
            justify-content: center;
            padding: 0;
            line-height: 1;
            pointer-events: auto;
        `;

        const mq = window.parent.matchMedia('(max-width: 767px)');
        const toggleVisibility = (e) => {
            btn.style.display = e.matches ? 'flex' : 'none';
        };
        toggleVisibility(mq);
        mq.addEventListener('change', toggleVisibility);

        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sb = getSidebarBtn();
            if (sb) sb.click();
        });

        doc.body.appendChild(btn);
    }

    injectHamburger();
    [200, 500, 1000].forEach(ms => setTimeout(injectHamburger, ms));
})();
</script>
""", height=0)

BG = "#13111a"; SURFACE = "#1a1727"; GRID = "#2d2640"; LABEL = "#6b5f8a"
C_SPOT = "#fc72ff"; C_TWAP = "#36d1b7"; C_ATTACK = "#ff6b6b"
C_PNL = "#36d1b7"; C_FEES = "#f5a623"; C_IL = "#ff6b6b"

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=LABEL, labelsize=8.5)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
        spine.set_linewidth(0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.4, linestyle="--", alpha=0.7)
    ax.grid(axis="x", visible=False)

def add_legend(ax):
    leg = ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor="#b8afd4", fontsize=8, framealpha=1)
    leg.get_frame().set_linewidth(0.5)

def section(label: str):
    st.markdown(
        f'<p style="font-size:0.95rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:#ede9f6 !important;'
        f'margin:2rem 0 0.75rem;border-bottom:1px solid #2d2640;'
        f'padding-bottom:0.5rem;">{label}</p>',
        unsafe_allow_html=True,
    )

def card_header(label: str):
    st.markdown(
        f'<p style="font-size:0.8rem;font-weight:600;color:#ede9f6 !important;'
        f'margin:0 0 0.5rem;">{label}</p>',
        unsafe_allow_html=True,
    )

TABLE_CSS = """
<style>
.opt-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.opt-table tr { border-bottom: 1px solid #2d2640; }
.opt-table tr:last-child { border-bottom: none; }
.opt-table td { padding: 0.45rem 0.6rem; color: #ede9f6; }
.opt-table td:first-child { color: #6b5f8a; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; width: 55%; }
.opt-wrap { background: #1a1727; border: 1px solid #2d2640; border-radius: 12px; overflow: hidden; }
</style>
"""

def fmt_table(data: dict) -> str:
    rows = ""
    for k, v in data.items():
        if isinstance(v, bool):
            val = "yes" if v else "no"
        elif isinstance(v, float):
            val = f"{v:.4f}"
        else:
            val = str(v)
        rows += f"<tr><td>{k}</td><td>{val}</td></tr>"
    return f'{TABLE_CSS}<div class="opt-wrap"><table class="opt-table">{rows}</table></div>'

col_header, col_toggle = st.columns([4, 1])

with col_header:
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#fc72ff !important;margin:0 0 0.35rem;">Research Prototype</p>
      <h1 style="font-size:1.9rem;margin:0 0 0.35rem;" class="title-gradient">
        AMM Options Engine
      </h1>
      <p style="font-size:0.875rem;color:#6b5f8a !important;margin:0;">
        AMM-derived pricing. No external oracles. Adversarial simulation.
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    if st.button("Toggle Parameters", use_container_width=True):
        st.session_state.show_params = not st.session_state.show_params

with st.sidebar:
    if st.session_state.show_params:
        st.markdown("### Parameters")
        st.session_state.n_steps = st.number_input(
            "Timesteps", 50, 300, st.session_state.n_steps, step=10)
        st.session_state.twap_window = st.number_input(
            "TWAP window", 2, 50, st.session_state.twap_window, step=1)
        st.session_state.initial_price = st.number_input(
            "Initial spot price (Y per X)", 100, 10_000,
            st.session_state.initial_price, step=50)
        st.session_state.attack_size = st.number_input(
            "Attack size (Y)", 10_000, 1_000_000,
            st.session_state.attack_size, step=10_000)
        st.session_state.run_attack = st.toggle(
            "Adversarial attack", value=st.session_state.run_attack)
        st.session_state.seed = st.number_input(
            "Seed", 0, 9999, st.session_state.seed)
        st.divider()
        if st.button("Run Simulation", use_container_width=True):
            st.session_state.run_btn = True

if st.session_state.run_btn:
    st.session_state.run_btn = False

    _initial_price = int(st.session_state.initial_price)
    _n_steps       = int(st.session_state.n_steps)
    _twap_window   = int(st.session_state.twap_window)
    _attack_size   = float(st.session_state.attack_size)
    _run_attack    = bool(st.session_state.run_attack)
    _seed          = int(st.session_state.seed)

    reserve_x = 1000.0
    reserve_y = float(_initial_price) * reserve_x

    cfg = SimulationConfig(
        n_steps=_n_steps,
        seed=_seed,
        initial_reserve_x=reserve_x,
        initial_reserve_y=reserve_y,
        twap_window=_twap_window,
        attack_size_y=_attack_size,
        run_attack=_run_attack,
    )

    with st.spinner("Running..."):
        sim = Simulation(cfg)
        df  = sim.run()

    attack_result = sim.attacker.result if sim.attacker else None
    st.session_state.results = dict(
        df=df,
        exploit=compute_exploitability(attack_result),
        lp_m=compute_lp_metrics(sim.lp_accounting.positions["lp_0"]),
        stability=compute_market_stability(df),
        twap_window=_twap_window,
        attack_step=cfg.attack_step,
        run_attack=_run_attack,
        attacker_option=(
            sim.attacker.option
            if sim.attacker and hasattr(sim.attacker, "option")
            else None
        ),
        entry_spot=reserve_y / reserve_x,
        option_expiry_offset=cfg.option_expiry_offset,
        initial_price=_initial_price,
    )
    st.session_state.has_run = True

if st.session_state.has_run and st.session_state.results:
    r         = st.session_state.results
    df        = r["df"]
    exploit   = r["exploit"]
    lp_m      = r["lp_m"]
    stability = r["stability"]
    profit    = exploit.get("attacker_gross_profit", 0)

    section("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attacker Profit (Y)", f"{profit:,.2f}",
              delta="exploitable" if profit > 0 else "not exploitable")
    c2.metric("LP Net PnL (Y)",   f"{lp_m['net_pnl']:,.2f}")
    c3.metric("IL Fraction",      f"{lp_m['il_fraction']:.4f}")
    c4.metric("Price Volatility", f"{stability['price_volatility']:.4f}")

    section("AMM Price")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df["step"], df["spot_price"], color=C_SPOT, lw=1.4, label="spot")
    ax.plot(df["step"], df["twap"], color=C_TWAP, lw=1.6,
            linestyle="--", label=f"twap w={r['twap_window']}")
    if r["run_attack"]:
        ax.axvline(r["attack_step"], color=C_ATTACK, lw=0.9, linestyle=":", label="attack")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("step", fontsize=8.5)
    ax.set_ylabel("Y / X", fontsize=8.5)
    style_ax(ax, fig)
    add_legend(ax)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)

    section("LP Performance")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df["step"], df["lp_net_pnl"],  color=C_PNL,  lw=1.4, label="net PnL")
    ax2.plot(df["step"], df["lp_fees"],     color=C_FEES, lw=1.0, linestyle="--", label="fees earned")
    ax2.plot(df["step"], df["lp_il_value"], color=C_IL,   lw=1.0, linestyle=":", label="IL value")
    ax2.axhline(0, color=GRID, lw=0.6)
    ax2.set_xlabel("step", fontsize=8.5)
    ax2.set_ylabel("Y tokens", fontsize=8.5)
    style_ax(ax2, fig2)
    add_legend(ax2)
    fig2.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)

    section("Detail")
    ca, cb = st.columns(2)
    with ca:
        card_header("Attack Summary")
        if exploit.get("attack_occurred"):
            display = {k: v for k, v in exploit.items() if k != "attack_occurred"}
            st.markdown(fmt_table(display), unsafe_allow_html=True)
        else:
            st.info("No attack run.")
    with cb:
        card_header("Market Stability")
        st.markdown(fmt_table(stability), unsafe_allow_html=True)

    if r["attacker_option"] is not None:
        opt = r["attacker_option"]
        pricing = compute_pricing_error(
            simulated_payoff=opt.payoff,
            strike=opt.strike,
            entry_spot=r["entry_spot"],
            sigma=0.3,
            steps_to_expiry=r["option_expiry_offset"],
        )
        cp, _ = st.columns(2)
        with cp:
            card_header("Pricing Error vs Black-Scholes")
            st.markdown(fmt_table(pricing), unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<p style="font-size:0.8rem;font-weight:600;color:#ede9f6 !important;'
        'margin:0 0 0.4rem;">Raw Simulation Log</p>',
        unsafe_allow_html=True,
    )
    with st.expander("", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
<div style="padding:1rem 0 2rem;">
  <p style="color:#6b5f8a !important;font-size:0.875rem;line-height:1.8;">
    Configure parameters in the sidebar and click <strong>Run Simulation</strong> to begin.
  </p>
</div>
""", unsafe_allow_html=True)