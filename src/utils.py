import base64, os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

def to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def make_html(charts, title, kpis=None):
    """charts = list of (title_str, matplotlib_fig)"""
    kpi_html = ''
    if kpis:
        kpi_html = '<div class="kpis">'
        for label, val in kpis:
            kpi_html += f'<div class="kpi"><b>{val}</b><span>{label}</span></div>'
        kpi_html += '</div>'

    cards = ''
    for name, fig in charts:
        cards += f'<div class="card"><p>{name}</p><img src="data:image/png;base64,{to_base64(fig)}"/></div>'

    return f"""<!DOCTYPE html><html><head><title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f6f9;margin:0;padding:24px;color:#222}}
h1{{color:#1e3a5f;border-left:4px solid #3b82f6;padding-left:10px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
.kpi{{background:#3b82f6;color:#fff;border-radius:8px;padding:16px 22px;text-align:center}}
.kpi b{{display:block;font-size:1.5rem}}
.kpi span{{font-size:.8rem;opacity:.85}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}}
.card{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.card p{{font-weight:600;color:#1e3a5f;margin:0 0 8px}}
.card img{{width:100%}}
</style></head><body>
<h1>{title}</h1>{kpi_html}<div class="grid">{cards}</div>
<p style="margin-top:24px;color:#999;font-size:.8rem">Jay Desai | jayd409@gmail.com</p>
</body></html>"""
