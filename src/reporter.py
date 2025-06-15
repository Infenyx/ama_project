import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from openai import OpenAI


def generate_client_report(
    client_id,
    master_df,
    output_path=None,
    risk_free_rate=0.02,
    openai_api_key=None,
    gpt_model="gpt-4o"
):
    """
    Generates a PDF performance report for a specific client, including automated chart interpretations via GPT.

    Parameters:
    - client_id: str
    - master_df: pandas.DataFrame with columns ['client','date','asset','quantity','price_usd']
    - output_path: str (defaults to client_report_{client_id}.pdf)
    - risk_free_rate: float (annualized)
    - openai_api_key: str (required for GPT analyses)
    - gpt_model: str (GPT model to use for interpretations)

    Returns:
    - output_path: path of the generated PDF
    """
    if not openai_api_key:
        raise ValueError("An OpenAI API key is required for automated interpretations.")
    client = OpenAI(api_key=openai_api_key)

    # Fonts & theme
    title_font = {'fontname': 'Times New Roman', 'fontsize': 24, 'fontweight': 'bold', 'color': '#2E4053'}
    subtitle_font = {'fontname': 'Times New Roman', 'fontsize': 15, 'color': '#34495E'}
    regular_font = {'fontname': 'DejaVu Sans', 'fontsize': 13, 'color': '#222'}
    table_font = {'fontname': 'DejaVu Sans', 'fontsize': 12}
    sns.set_theme(style='whitegrid', font_scale=1.2)
    palette = sns.color_palette("tab10", 10)

    # Data prep
    df = master_df.copy()
    required = ['client','date','asset','quantity','price_usd']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[df['client']==client_id]
    if df.empty:
        raise ValueError(f"No data for client {client_id}")
    df['date'] = pd.to_datetime(df['date'])
    df['value_usd'] = df['quantity'] * df['price_usd']
    df = df.sort_values('date')

    # Metrics
    daily_value = df.groupby('date')['value_usd'].sum()
    daily_returns = daily_value.pct_change().dropna().clip(-0.5, 0.5)
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    twr = (1 + daily_returns).prod() - 1

    trading_days = 252
    ann_vol = daily_returns.std() * np.sqrt(trading_days)
    rf_daily = (1 + risk_free_rate) ** (1/trading_days) - 1
    sharpe = (daily_returns.mean() - rf_daily) / daily_returns.std() * np.sqrt(trading_days)
    downside = np.minimum(daily_returns, 0)
    sortino = (daily_returns.mean() - rf_daily) / (downside.std() * np.sqrt(trading_days)) if downside.std()>0 else np.nan
    cum = (1 + daily_returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd = drawdown.min()

    latest = df['date'].max()
    holdings = df[df['date']==latest].groupby('asset')['value_usd'].sum()
    top10 = holdings.nlargest(10)
    top10_pct = top10.sum() / holdings.sum() if holdings.sum()>0 else np.nan
    hhi = (holdings/holdings.sum()).pow(2).sum() if holdings.sum()>0 else np.nan

    if not output_path:
        # Default output path inside 'client_reports' folder in the project root
        reports_dir = os.path.join(os.getcwd(), "client_reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"client_report_{client_id}.pdf")


    def analyze_and_embed(fig, prompt_text="Please analyze this chart and provide insights."):
        # Always use standardized size
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        resp = client.chat.completions.create(
            model=gpt_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }]
        )
        analysis = resp.choices[0].message.content
        
        # Use same page size as the charts
        fig2, ax2 = plt.subplots(figsize=(11, 8.5))
        ax2.axis('off')
        fig2.patch.set_facecolor('#F7F9FB')
        fig2.text(0.06, 0.93, "Automated Interpretation", fontname="Times New Roman",
                fontsize=22, fontweight="bold", color="#2E4053")
        fig2.text(0.06, 0.88, prompt_text, fontname="DejaVu Sans", fontsize=14, color="#34495E")
        y = 0.83
        line_spacing = 0.042  # Slightly tighter
        font_size = 13  # Slightly smaller for longer text
        for paragraph in analysis.split('\n'):
            for line in textwrap.wrap(paragraph, width=115):
                fig2.text(0.06, y, line, fontname="DejaVu Sans", fontsize=font_size, color="#2E2E2E")
                y -= line_spacing
            y -= line_spacing / 2  # Small gap after paragraph
            if y < 0.08:
                break  # Prevent overflow
        plt.tight_layout()
        return fig2



    # Create PDF
    with PdfPages(output_path) as pdf:
        figsize_standard = (11, 8.5)
        # Executive Summary Page
        fig, ax = plt.subplots(figsize=figsize_standard)
        ax.axis('off')
        fig.patch.set_facecolor("#000000")
        # Soft background card
        bbox = FancyBboxPatch((0.02, 0.1), 0.96, 0.78,
                            boxstyle="round,pad=0.04", linewidth=1.8,
                            edgecolor='#2E4053', facecolor='white', transform=fig.transFigure, zorder=1)
        fig.patches.append(bbox)
        # Headings
        fig.text(0.06, 0.89, f"Performance Report: {client_id}", **title_font)
        fig.text(0.06, 0.84, f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", **subtitle_font)
        fig.text(0.06, 0.79, f"Period: {df['date'].min().date()} to {latest.date()}", **subtitle_font)
        # Table data
        labels = [
            "Cumulative Return", "Time-Weighted Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown",
            "Top 10 Holdings %", "Herfindahl Index"
        ]
        values = [
            f"{cumulative_returns.iloc[-1]:.2%}", f"{twr:.2%}", f"{ann_vol:.2%}",
            f"{sharpe:.2f}", f"{sortino:.2f}", f"{max_dd:.2%}",
            f"{top10_pct:.2%}", f"{hhi:.4f}"
        ]
        # Pandas DataFrame to control table appearance
        metrics_df = pd.DataFrame({'Metric': labels, 'Value': values})
        # Table position and style
        # Inside the executive summary page creation
        # After the headings and before the footer, drop the entire ax.table(...) section
        # and insert this instead:

        # --- Manual Metrics Grid ---
        labels = [
            "Cumulative Return", "Time-Weighted Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown",
            "Top 10 Holdings %", "Herfindahl Index"
        ]
        values = [
            f"{cumulative_returns.iloc[-1]:.2%}", f"{twr:.2%}", f"{ann_vol:.2%}",
            f"{sharpe:.2f}", f"{sortino:.2f}", f"{max_dd:.2%}",
            f"{top10_pct:.2%}", f"{hhi:.4f}"
        ]

        # Starting Y position and spacing
        start_y = 0.72
        dy = 0.055

        # Column X positions
        x_label = 0.18
        x_value = 0.60

        for i, (lab, val) in enumerate(zip(labels, values)):
            y = start_y - i * dy
            fig.text(x_label, y, lab + ":", fontname="DejaVu Sans", fontsize=14, color="#2E4053")
            fig.text(x_value, y, val, fontname="DejaVu Sans", fontsize=14, color="#1a1a1a")



        # Footer
        fig.text(0.06, 0.12, "Confidential – For Client Use Only", fontsize=11, color="#8395a7", style='italic')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # [Charts and interpretations remain unchanged from previous implementation]
        
        # 1) Cumulative Returns
        fig_cr, ax_cr = plt.subplots(figsize=figsize_standard)
        fig_cr.patch.set_facecolor('#F7F9FB')
        ax_cr.set_facecolor('white')
        ax_cr.plot(cumulative_returns.index, cumulative_returns.values, color='#3D5AFE', linewidth=2.5)
        ax_cr.set_title("Cumulative Portfolio Return", **title_font)
        ax_cr.set_ylabel("Cumulative Return", **regular_font)
        ax_cr.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
        ax_cr.spines['top'].set_visible(False)
        ax_cr.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_cr)
        pdf.savefig(analyze_and_embed(fig_cr))

        # 2) Monthly Returns
        fig_mr, ax_mr = plt.subplots(figsize=figsize_standard)
        fig_mr.patch.set_facecolor('#F7F9FB')
        ax_mr.set_facecolor('white')
        ax_mr.bar(monthly_returns.index.strftime('%b %Y'), monthly_returns.values, color=palette)
        ax_mr.set_title("Monthly Portfolio Returns", **title_font)
        ax_mr.set_ylabel("Return", **regular_font)
        ax_mr.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
        plt.setp(ax_mr.get_xticklabels(), rotation=35, ha='right')
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_mr)
        pdf.savefig(analyze_and_embed(fig_mr, "Analyze the monthly returns chart and explain key trends."))

        # 3) Top 10 Holdings Pie
        figsize_standard = (11, 8.5)
        fig_pie, ax_pie = plt.subplots(figsize=figsize_standard)
        fig_pie.patch.set_facecolor('#F7F9FB')
        ax_pie.set_facecolor('white')
        wedges, texts, autotexts = ax_pie.pie(top10, labels=top10.index,
                                              autopct='%1.1f%%', startangle=120,
                                              colors=palette,
                                              wedgeprops={'edgecolor':'white','linewidth':1.2})
        ax_pie.set_title(f"Top 10 Holdings Allocation ({latest.date()})", **title_font)
        ax_pie.axis('equal')
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_pie)
        pdf.savefig(analyze_and_embed(fig_pie, "Provide insights on the top 10 holdings allocation pie chart."))

        # 4) Drawdown Curve
        fig_dd, ax_dd = plt.subplots(figsize=figsize_standard)
        fig_dd.patch.set_facecolor('#F7F9FB')
        ax_dd.set_facecolor('white')
        ax_dd.plot(drawdown.index, drawdown.values, color='#D7263D', linewidth=2.2)
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color='#FBB1B1', alpha=0.6)
        ax_dd.set_title("Portfolio Drawdown (%)", **title_font)
        ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
        ax_dd.spines['top'].set_visible(False)
        ax_dd.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_dd)
        pdf.savefig(analyze_and_embed(fig_dd, "Please analyze the portfolio drawdown chart and explain its key takeaways."))

        # 5) Rolling 21-Day Volatility
        roll_vol = daily_returns.rolling(21).std() * np.sqrt(trading_days)
        fig_rv, ax_rv = plt.subplots(figsize=figsize_standard)
        fig_rv.patch.set_facecolor('#F7F9FB')
        ax_rv.set_facecolor('white')
        ax_rv.plot(roll_vol.index, roll_vol.values, color='#00897B', linewidth=2.2)
        ax_rv.set_title("Rolling 21-Day Annualized Volatility", **title_font)
        ax_rv.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
        ax_rv.spines['top'].set_visible(False)
        ax_rv.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_rv)
        pdf.savefig(analyze_and_embed(fig_rv, "Analyze the rolling 21-day volatility chart and highlight any volatility clusters."))

        # 6) Portfolio Value Over Time
        fig_pv, ax_pv = plt.subplots(figsize=figsize_standard)
        fig_pv.patch.set_facecolor('#F7F9FB')
        ax_pv.set_facecolor('white')
        ax_pv.plot(daily_value.index, daily_value.values, color='#2E4053', linewidth=2.5)
        ax_pv.set_title("Portfolio Value Over Time (USD)", **title_font)
        ax_pv.set_ylabel("Portfolio Value ($)", **regular_font)
        ax_pv.spines['top'].set_visible(False)
        ax_pv.spines['right'].set_visible(False)
        plt.tight_layout(pad=2.0)
        pdf.savefig(fig_pv)
        pdf.savefig(analyze_and_embed(fig_pv, "Provide insights on the portfolio value over time chart."))

    print(f"Report for {client_id} saved to {output_path}")
    return output_path

# Example usage:
# master_df = pd.read_csv('master.csv')
# generate_client_report('Client_2', master_df, openai_api_key="YOUR_API_KEY_HERE")