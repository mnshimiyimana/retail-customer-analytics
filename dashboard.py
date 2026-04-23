"""
Retail Customer Analytics — Interactive Dashboard
Built with Plotly Dash + Dash Bootstrap Components
Run:  python3 dashboard.py   then open  http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N_ORDERS, N_CUSTOMERS = 18_000, 4_200

categories = {
    'Electronics':    {'products': ['Wireless Headphones','Smartwatch','USB-C Hub','Webcam','Keyboard'],
                       'price_range': (25, 280), 'weight': 0.22},
    'Clothing':       {'products': ['Running Jacket','Denim Jeans','Casual T-Shirt','Hooded Sweatshirt','Chinos'],
                       'price_range': (15, 120), 'weight': 0.25},
    'Home & Kitchen': {'products': ['Air Fryer','Coffee Maker','Blender','Smart Bulb','Bamboo Cutting Board'],
                       'price_range': (12, 180), 'weight': 0.20},
    'Books':          {'products': ['Data Science Handbook','Python Crash Course','Atomic Habits','Sapiens','Fiction Novel'],
                       'price_range': (8,  35),  'weight': 0.13},
    'Sports':         {'products': ['Yoga Mat','Resistance Bands','Water Bottle','Foam Roller','Jump Rope'],
                       'price_range': (10, 90),  'weight': 0.12},
    'Beauty':         {'products': ['Face Serum','Moisturiser','SPF50 Sunscreen','Eye Cream','Lip Balm'],
                       'price_range': (8,  65),  'weight': 0.08},
}
regions   = ['London','South East','Midlands','North West','Scotland','Yorkshire','Wales','North East']
channels  = ['Organic Search','Paid Search','Email','Social Media','Direct','Referral']
region_w  = [0.24,0.18,0.16,0.14,0.09,0.08,0.06,0.05]
channel_w = [0.28,0.22,0.20,0.15,0.10,0.05]

all_dates   = pd.date_range('2024-01-01', '2025-12-31', freq='D')
day_of_year = np.array([d.timetuple().tm_yday for d in all_dates])
seasonal_w  = 1 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
holiday_b   = np.where(
    (np.array([d.month for d in all_dates]) == 11) |
    (np.array([d.month for d in all_dates]) == 12), 1.8, 1.0)
date_w = seasonal_w * holiday_b
date_w /= date_w.sum()
order_dates = np.random.choice(all_dates, size=N_ORDERS, replace=True, p=date_w)

customer_freq = np.clip(np.random.zipf(1.5, N_CUSTOMERS), 1, 50)
cust_p   = customer_freq / customer_freq.sum()
cust_ids = [f'C{str(i).zfill(5)}' for i in range(N_CUSTOMERS)]
order_customers = np.random.choice(cust_ids, size=N_ORDERS, replace=True, p=cust_p)

cat_names   = list(categories.keys())
cat_weights = [categories[c]['weight'] for c in cat_names]
order_cats  = np.random.choice(cat_names, size=N_ORDERS, replace=True, p=cat_weights)
order_products, order_prices = [], []
for cat in order_cats:
    lo, hi = categories[cat]['price_range']
    order_products.append(np.random.choice(categories[cat]['products']))
    order_prices.append(round(np.random.uniform(lo, hi), 2))
order_qty = np.random.choice([1,1,1,2,2,3,4,5], size=N_ORDERS,
                              p=[0.45,0.25,0.15,0.07,0.04,0.02,0.01,0.01])

df = pd.DataFrame({
    'order_id':    [f'ORD{str(i+1000).zfill(6)}' for i in range(N_ORDERS)],
    'customer_id': order_customers,
    'order_date':  pd.to_datetime(order_dates),
    'category':    order_cats,
    'product':     order_products,
    'quantity':    order_qty,
    'unit_price':  order_prices,
    'region':      np.random.choice(regions, size=N_ORDERS, replace=True, p=region_w),
    'channel':     np.random.choice(channels, size=N_ORDERS, replace=True, p=channel_w),
})
df['revenue']     = (df['unit_price'] * df['quantity']).round(2)
df['year']        = df['order_date'].dt.year
df['month']       = df['order_date'].dt.month
df['day_of_week'] = df['order_date'].dt.day_name()
df['year_month']  = df['order_date'].dt.to_period('M')

# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE-COMPUTED DATASETS
# ══════════════════════════════════════════════════════════════════════════════

monthly = df.groupby('year_month').agg(
    revenue=('revenue','sum'),
    orders=('order_id','count'),
    customers=('customer_id','nunique'),
    aov=('revenue','mean')
).reset_index()
monthly['year_month_str'] = monthly['year_month'].astype(str)
monthly['mom_growth']     = monthly['revenue'].pct_change() * 100

# RFM + K-Means
snapshot = df['order_date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg(
    recency=('order_date',  lambda x: (snapshot - x.max()).days),
    frequency=('order_id',  'count'),
    monetary=('revenue',    'sum')
).reset_index()

def quintile_score(s, rev=False):
    labels = [5,4,3,2,1] if rev else [1,2,3,4,5]
    try:    return pd.qcut(s, q=5, labels=labels, duplicates='drop').astype(int)
    except: return pd.cut(s, bins=5, labels=labels).astype(int)

rfm['R'] = quintile_score(rfm['recency'], rev=True)
rfm['F'] = quintile_score(rfm['frequency'])
rfm['M'] = quintile_score(rfm['monetary'])

X  = StandardScaler().fit_transform(rfm[['recency','frequency','monetary']])
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
rfm['cluster'] = km.labels_

composite = (
    -rfm.groupby('cluster')['recency'].mean()
    + rfm.groupby('cluster')['frequency'].mean() * 30
    + rfm.groupby('cluster')['monetary'].mean() * 0.01
)
rank = composite.rank()
label_map = {
    rank.idxmax():              'Champions',
    rank.nlargest(2).index[-1]: 'Loyal Customers',
    rank.nsmallest(2).index[-1]:'At-Risk Customers',
    rank.idxmin():              'Lost / Inactive',
}
rfm['segment'] = rfm['cluster'].map(label_map)

# Cohort retention
first_order = df.groupby('customer_id')['year_month'].min().rename('cohort_month')
df2 = df.join(first_order, on='customer_id')
df2['cohort_index'] = (
    (df2['year_month'].dt.year  - df2['cohort_month'].dt.year) * 12 +
    (df2['year_month'].dt.month - df2['cohort_month'].dt.month)
)
cohort_pivot = (
    df2.groupby(['cohort_month','cohort_index'])['customer_id']
       .nunique().reset_index()
       .pivot_table(index='cohort_month', columns='cohort_index', values='customer_id')
)
retention = cohort_pivot.divide(cohort_pivot[0], axis=0).round(3) * 100
retention.index = retention.index.astype(str)

# Forecast
monthly['t'] = np.arange(len(monthly))
for mo in range(1, 13):
    monthly[f'mo_{mo}'] = (monthly['year_month'].dt.month == mo).astype(int)
feat = ['t'] + [f'mo_{mo}' for mo in range(1, 13)]
mdl = LinearRegression().fit(monthly[feat], monthly['revenue'])
monthly['fitted'] = mdl.predict(monthly[feat])
resid_std = np.std(monthly['revenue'] - monthly['fitted'])

last_t  = monthly['t'].max()
last_m  = monthly['year_month'].max()
fut_months = pd.period_range(start=last_m + 1, periods=6, freq='M')
fut = pd.DataFrame({'year_month': fut_months, 't': np.arange(last_t+1, last_t+7)})
for mo in range(1, 13):
    fut[f'mo_{mo}'] = (fut['year_month'].dt.month == mo).astype(int)
fut['forecast']       = mdl.predict(fut[feat])
fut['lower']          = fut['forecast'] - 1.96 * resid_std
fut['upper']          = fut['forecast'] + 1.96 * resid_std
fut['year_month_str'] = fut['year_month'].astype(str)

# KPIs
total_rev  = df['revenue'].sum()
rev_2024   = df[df['year']==2024]['revenue'].sum()
rev_2025   = df[df['year']==2025]['revenue'].sum()
yoy        = (rev_2025 - rev_2024) / rev_2024 * 100
n_cust     = df['customer_id'].nunique()
avg_ov     = df['revenue'].mean()
champions  = rfm[rfm['segment'] == 'Champions']
champ_pct  = champions['monetary'].sum() / rfm['monetary'].sum() * 100


# ══════════════════════════════════════════════════════════════════════════════
# 3. THEME COLOURS  (used by Plotly figures — CSS handles the rest)
# ══════════════════════════════════════════════════════════════════════════════
C = {
    'bg':    '#0b0d14', 'surface': '#13161f', 'surface2': '#1c2030',
    'border':'#252a3d', 'text':    '#e2e5f0', 'muted':    '#7278a0',
    'i':     '#6366f1', 'c':       '#22d3ee', 'e':        '#10b981',
    'a':     '#f59e0b', 'r':       '#f43f5e', 'v':        '#a855f7',
}
SEG_COLORS = {
    'Champions':        '#10b981',
    'Loyal Customers':  '#6366f1',
    'At-Risk Customers':'#f59e0b',
    'Lost / Inactive':  '#f43f5e',
}
CAT_COLORS = px.colors.qualitative.Bold

def T(fig, m=None, xa=None, ya=None):
    base_m  = dict(l=42, r=18, t=36, b=36)
    base_xa = dict(gridcolor=C['border'], zerolinecolor='rgba(0,0,0,0)',
                   linecolor=C['border'])
    base_ya = dict(gridcolor=C['border'], zerolinecolor='rgba(0,0,0,0)',
                   linecolor=C['border'])
    if m:  base_m.update(m)
    if xa: base_xa.update(xa)
    if ya: base_ya.update(ya)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['text'], family='Inter, system-ui, sans-serif', size=11.5),
        hoverlabel=dict(bgcolor='#1c2030', font_color=C['text'],
                        bordercolor=C['border'], font_size=12),
        margin=base_m, xaxis=base_xa, yaxis=base_ya,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_revenue_trend(cat=None, yr=None):
    data = df.copy()
    if cat and cat != 'All': data = data[data['category'] == cat]
    if yr  and yr  != 'All': data = data[data['year'] == int(yr)]
    m = (data.groupby(data['order_date'].dt.to_period('M'))
             .agg(revenue=('revenue','sum')).reset_index())
    m['ds']  = m['order_date'].dt.to_timestamp()
    m['rol'] = m['revenue'].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m['ds'], y=m['revenue'], name='Monthly Revenue',
        marker=dict(color=C['i'], opacity=0.55,
                    line=dict(color=C['i'], width=0)),
        hovertemplate='<b>%{x|%b %Y}</b><br>£%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=m['ds'], y=m['rol'], name='3-mo Avg',
        line=dict(color=C['c'], width=2.5),
        hovertemplate='<b>%{x|%b %Y}</b><br>Avg £%{y:,.0f}<extra></extra>'))
    T(fig)
    fig.update_layout(yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=1.0, x=0))
    return fig

def fig_mom_growth():
    colors = [C['e'] if v >= 0 else C['r'] for v in monthly['mom_growth']]
    fig = go.Figure(go.Bar(x=monthly['year_month_str'], y=monthly['mom_growth'],
        marker_color=colors, opacity=0.85,
        hovertemplate='<b>%{x}</b><br>%{y:+.1f}%<extra></extra>'))
    fig.add_hline(y=0, line_dash='dot', line_color=C['muted'], line_width=1)
    T(fig)
    fig.update_layout(xaxis_tickangle=-45, yaxis_ticksuffix='%',
                      showlegend=False)
    return fig

def fig_category_sunburst():
    d = df.groupby(['category','product'])['revenue'].sum().reset_index()
    fig = px.sunburst(d, path=['category','product'], values='revenue',
        color='category', color_discrete_sequence=CAT_COLORS)
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>£%{value:,.0f} · %{percentParent:.0%}<extra></extra>',
        textfont_size=11, insidetextorientation='radial')
    T(fig, m=dict(l=4, r=4, t=4, b=4))
    return fig

def fig_channel_perf():
    ch = df.groupby('channel').agg(
        revenue=('revenue','sum'), orders=('order_id','count')).reset_index()
    ch['aov'] = ch['revenue'] / ch['orders']
    ch.sort_values('revenue', ascending=True, inplace=True)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=['Revenue by Channel', 'Avg Order Value'])
    fig.add_trace(go.Bar(y=ch['channel'], x=ch['revenue'], orientation='h',
        marker=dict(color=ch['revenue'],
                    colorscale=[[0,'#1c2030'],[0.5,C['i']],[1,C['c']]]),
        text=[f'£{v:,.0f}' for v in ch['revenue']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>£%{x:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Bar(y=ch['channel'], x=ch['aov'], orientation='h',
        marker_color=C['a'], opacity=0.8,
        text=[f'£{v:.0f}' for v in ch['aov']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>AOV £%{x:.2f}<extra></extra>'), row=1, col=2)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['text'], family='Inter, system-ui, sans-serif', size=11.5),
        hoverlabel=dict(bgcolor='#1c2030', font_color=C['text'], bordercolor=C['border']),
        margin=dict(l=10, r=18, t=36, b=36),
        showlegend=False)
    fig.update_xaxes(gridcolor=C['border'], zerolinecolor='rgba(0,0,0,0)',
                     tickprefix='£', tickformat=',.0f')
    fig.update_yaxes(gridcolor=C['border'], zerolinecolor='rgba(0,0,0,0)')
    fig.update_annotations(font=dict(color=C['text'], size=12))
    return fig

def fig_rfm_scatter(seg=None):
    data = rfm.copy()
    if seg and seg != 'All': data = data[data['segment'] == seg]
    fig = px.scatter(data, x='frequency', y='monetary', color='segment',
        size='monetary', size_max=18, color_discrete_map=SEG_COLORS, opacity=0.6,
        labels={'frequency':'Order Frequency','monetary':'Lifetime Value (£)'},
        custom_data=['customer_id','recency','monetary','segment'])
    fig.update_traces(
        hovertemplate='<b>%{customdata[3]}</b><br>'
                      '%{customdata[0]}<br>'
                      'Orders: %{x} · LTV: £%{y:,.0f}<br>'
                      'Last order: %{customdata[1]}d ago<extra></extra>')
    T(fig)
    fig.update_layout(yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        legend=dict(bgcolor='rgba(0,0,0,0)', title=''))
    return fig

def fig_rfm_treemap():
    seg_rev = rfm.groupby('segment').agg(
        customers=('customer_id','count'), total=('monetary','sum'),
        avg_ltv=('monetary','mean'), avg_freq=('frequency','mean')).reset_index()
    fig = px.treemap(seg_rev, path=['segment'], values='total',
        color='avg_ltv',
        color_continuous_scale=['#13161f','#6366f1','#22d3ee'],
        custom_data=['customers','avg_ltv','avg_freq'])
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>'
                      'Customers: %{customdata[0]:,}<br>'
                      'Avg LTV: £%{customdata[1]:,.0f}<br>'
                      'Avg Orders: %{customdata[2]:.1f}<extra></extra>',
        textfont_size=12, marker_line_width=2,
        marker_line_color=C['bg'])
    T(fig, m=dict(l=4, r=4, t=4, b=4))
    fig.update_layout(coloraxis_showscale=False)
    return fig

def fig_cohort_heatmap():
    ret = retention.iloc[:, :13].copy()
    fig = go.Figure(go.Heatmap(
        z=ret.values, x=[f'Mo {i}' for i in ret.columns], y=ret.index.tolist(),
        colorscale=[[0,'#0b0d14'],[0.25,'#252a3d'],[0.55,'#6366f1'],[1,'#22d3ee']],
        zmin=0, zmax=100,
        text=[[f'{v:.0f}%' if not np.isnan(v) else '' for v in row] for row in ret.values],
        texttemplate='%{text}', textfont=dict(size=8.5),
        hovertemplate='Cohort: %{y}<br>%{x}<br>Retention: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='%', ticksuffix='%', len=0.8,
                      tickfont=dict(color=C['text']),
                      title_font=dict(color=C['muted']))))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['text'], family='Inter, system-ui, sans-serif', size=11),
        hoverlabel=dict(bgcolor='#1c2030', font_color=C['text'], bordercolor=C['border']),
        margin=dict(l=42, r=18, t=16, b=36),
        xaxis=dict(side='bottom', gridcolor='rgba(0,0,0,0)',
                   linecolor=C['border']),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor=C['border'], autorange='reversed'))
    return fig

def fig_retention_curve():
    avg = retention.mean()
    x, y = list(range(13)), avg.values[:13]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
        fill='tozeroy', fillcolor='rgba(99,102,241,0.12)',
        line=dict(color=C['i'], width=2.5),
        marker=dict(size=7, color=C['c'], line=dict(color=C['bg'], width=2)),
        hovertemplate='Month %{x}<br>Avg Retention: %{y:.1f}%<extra></extra>'))
    T(fig)
    fig.update_layout(yaxis_ticksuffix='%', showlegend=False,
        xaxis=dict(tickmode='linear', dtick=1,
                   gridcolor=C['border'], zerolinecolor='rgba(0,0,0,0)'))
    return fig

def fig_forecast():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['year_month_str'], y=monthly['revenue'],
        name='Actual', marker=dict(color=C['i'], opacity=0.5),
        hovertemplate='<b>%{x}</b><br>£%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=monthly['year_month_str'], y=monthly['fitted'],
        name='Fit', line=dict(color=C['c'], width=1.8, dash='dot'),
        hovertemplate='<b>%{x}</b><br>Fit £%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=list(fut['year_month_str']) + list(fut['year_month_str'])[::-1],
        y=list(fut['upper']) + list(fut['lower'])[::-1],
        fill='toself', fillcolor='rgba(245,158,11,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% PI', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fut['year_month_str'], y=fut['forecast'],
        name='Forecast', mode='lines+markers',
        line=dict(color=C['a'], width=2.5),
        marker=dict(size=8, color=C['a'], line=dict(color=C['bg'], width=2)),
        hovertemplate='<b>%{x}</b><br>Forecast £%{y:,.0f}<extra></extra>'))
    T(fig)
    fig.update_layout(
        yaxis_tickprefix='£', yaxis_tickformat=',.0f',
        xaxis_tickangle=-45,
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=1.0, x=0))
    return fig

def fig_region_rev():
    reg = df.groupby('region').agg(
        revenue=('revenue','sum'), customers=('customer_id','nunique'),
        orders=('order_id','count')).reset_index()
    reg['aov'] = reg['revenue'] / reg['orders']
    reg.sort_values('revenue', ascending=True, inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=reg['region'], x=reg['revenue'], orientation='h',
        marker=dict(color=reg['revenue'],
                    colorscale=[[0,'#1c2030'],[0.5,C['i']],[1,C['c']]]),
        text=[f'£{v:,.0f}' for v in reg['revenue']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>'))
    T(fig, m=dict(l=10, r=60, t=36, b=36))
    fig.update_layout(xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                      showlegend=False)
    return fig

def fig_top_products():
    top = df.groupby('product')['revenue'].sum().nlargest(12).reset_index()
    top.sort_values('revenue', inplace=True)
    fig = go.Figure(go.Bar(
        x=top['revenue'], y=top['product'], orientation='h',
        marker=dict(color=top['revenue'],
                    colorscale=[[0,'#252a3d'],[0.5,C['i']],[1,C['c']]]),
        text=[f'£{v:,.0f}' for v in top['revenue']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>£%{x:,.0f}<extra></extra>'))
    T(fig, m=dict(l=10, r=60, t=36, b=36))
    fig.update_layout(xaxis_tickprefix='£', xaxis_tickformat=',.0f',
                      showlegend=False)
    return fig

def fig_dow_heat():
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    piv = (df.groupby(['month','day_of_week'])['revenue'].mean()
             .reset_index()
             .pivot(index='day_of_week', columns='month', values='revenue')
             .reindex(dow_order))
    piv.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale=[[0,'#0b0d14'],[0.4,'#252a3d'],[0.7,C['i']],[1,C['c']]],
        text=[[f'£{v:,.0f}' for v in row] for row in piv.values],
        texttemplate='%{text}', textfont=dict(size=8.5),
        hovertemplate='<b>%{y}, %{x}</b><br>Avg £%{z:,.0f}<extra></extra>',
        colorbar=dict(len=0.9, tickprefix='£',
                      tickfont=dict(color=C['text']))))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['text'], family='Inter, system-ui, sans-serif', size=11),
        hoverlabel=dict(bgcolor='#1c2030', font_color=C['text'], bordercolor=C['border']),
        margin=dict(l=10, r=18, t=16, b=36),
        xaxis=dict(gridcolor='rgba(0,0,0,0)', side='bottom'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 5. UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi(value, label, sub, variant):
    return html.Div([
        html.Div(className=f'kpi-icon'),
        html.Div(value, className='kpi-value'),
        html.Div(label, className='kpi-label'),
        html.Div(sub,   className='kpi-sub'),
    ], className=f'kpi-card {variant}')

def chart(fig_id, fig=None, h=370):
    return html.Div([
        dcc.Graph(id=fig_id, figure=fig,
                  config={'displayModeBar':True,'displaylogo':False,
                          'modeBarButtonsToRemove':['select2d','lasso2d','toImage']},
                  style={'height':f'{h}px'})
    ], className='chart-card')

def section(title):
    return html.Div([
        html.Div(className='section-bar'),
        html.H5(title),
    ], className='section-heading')

def dropdown(id_, options, value):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
        style={'background':'#1c2030','color':'#000','border':'1px solid #252a3d',
               'borderRadius':'8px','fontSize':'12px'})

# ══════════════════════════════════════════════════════════════════════════════
# 6. SEGMENT SUMMARY TABLE DATA
# ══════════════════════════════════════════════════════════════════════════════
seg_table_data = (
    rfm.groupby('segment')
       .agg(customers=('customer_id','count'),
            avg_recency=('recency','mean'),
            avg_frequency=('frequency','mean'),
            avg_monetary=('monetary','mean'),
            total_revenue=('monetary','sum'))
       .assign(**{'rev_share': lambda x: x['total_revenue']/x['total_revenue'].sum()*100})
       .drop(columns='total_revenue')
       .reset_index().round(1).to_dict('records')
)

TABLE_STYLE = dict(
    style_table={'overflowX':'auto','borderRadius':'8px','overflow':'hidden'},
    style_header={'backgroundColor':'#1c2030','color':C['text'],
                  'fontWeight':'700','border':'1px solid #252a3d',
                  'fontSize':'0.76rem','fontFamily':'Inter, sans-serif',
                  'padding':'10px 14px'},
    style_data={'backgroundColor':'#13161f','color':C['text'],
                'border':'1px solid #252a3d','fontSize':'0.76rem',
                'fontFamily':'Inter, sans-serif','padding':'9px 14px'},
    style_data_conditional=[
        {'if':{'filter_query':'{segment} = "Champions"'},
         'color':'#10b981','fontWeight':'700'},
        {'if':{'filter_query':'{segment} = "Loyal Customers"'},
         'color':'#6366f1','fontWeight':'600'},
        {'if':{'filter_query':'{segment} = "At-Risk Customers"'},
         'color':'#f59e0b','fontWeight':'600'},
        {'if':{'filter_query':'{segment} = "Lost / Inactive"'},
         'color':'#f43f5e'},
        {'if':{'row_index':'odd'},'backgroundColor':'#1c2030'},
        {'if':{'state':'active'},'backgroundColor':'#252a3d',
         'border':'1px solid #303553'},
    ],
    sort_action='native',
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. APP
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap',
    ],
    title='Retail Analytics · Maureen T. N.',
    suppress_callback_exceptions=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def nav_item(icon, label, href):
    return dcc.Link(
        html.Div([
            html.Span(icon, className='nav-icon'),
            html.Span(label),
        ], className='nav-link'),
        href=href, style={'textDecoration':'none'})

sidebar = html.Div(id='sidebar', style={
    'position':'fixed','top':0,'left':0,'bottom':0,'width':'220px',
    'padding':'20px 14px','zIndex':200,'overflowY':'auto',
}, children=[
    # Brand
    html.Div([
        html.Div('📊', className='brand-logo'),
        html.Div([
            html.Div('Retail Analytics',
                     style={'color':'#e2e5f0','fontWeight':'800','fontSize':'0.9rem',
                            'letterSpacing':'-0.2px'}),
            html.Div('Maureen T. N.',
                     style={'color':'#7278a0','fontSize':'0.7rem','marginTop':'1px'}),
        ]),
    ], style={'display':'flex','alignItems':'center','gap':'10px',
              'marginBottom':'28px','paddingBottom':'20px',
              'borderBottom':'1px solid #252a3d'}),

    html.Div('Menu', className='nav-label'),
    nav_item('🏠', 'Overview',      '/'),
    nav_item('📈', 'Sales Trends',  '/trends'),
    nav_item('👥', 'Customers',     '/customers'),
    nav_item('🔄', 'Retention',     '/retention'),
    nav_item('🔮', 'Forecast',      '/forecast'),
    nav_item('🗺', 'Regional',      '/regional'),

    html.Div([
        html.Div('Dataset', style={'fontSize':'0.65rem','fontWeight':'700',
                                   'letterSpacing':'1px','color':'#3e4566',
                                   'textTransform':'uppercase','marginBottom':'8px'}),
        html.Div('Jan 2024 – Dec 2025', style={'color':'#7278a0','fontSize':'0.72rem'}),
        html.Div('18,000 transactions', style={'color':'#7278a0','fontSize':'0.72rem'}),
        html.Div('4,200 customers',     style={'color':'#7278a0','fontSize':'0.72rem'}),
    ], className='sidebar-footer'),
])

app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    html.Div(id='main-content', style={
        'marginLeft':'220px','padding':'28px 32px',
        'minHeight':'100vh',
    }),
], style={'background':'#0b0d14','fontFamily':'Inter, system-ui, sans-serif'})

# ══════════════════════════════════════════════════════════════════════════════
# 8. PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_header(title, subtitle):
    return html.Div([
        html.H2(title),
        html.P(subtitle),
    ], className='page-header')

# ── Overview ──────────────────────────────────────────────────────────────────
page_overview = html.Div([
    page_header('Executive Overview',
                'Jan 2024 – Dec 2025  ·  18,000 transactions  ·  4,200 customers'),

    # KPI row
    dbc.Row([
        dbc.Col(kpi(f'£{total_rev:,.0f}',  'Total Revenue',       'Jan 2024 – Dec 2025', 'indigo'),  md=2),
        dbc.Col(kpi(f'+{yoy:.1f}%',         'YoY Growth',          '2025 vs 2024',         'emerald'), md=2),
        dbc.Col(kpi(f'{n_cust:,}',           'Unique Customers',    'active buyers',         'cyan'),   md=2),
        dbc.Col(kpi(f'£{avg_ov:.2f}',        'Avg Order Value',     'per transaction',       'amber'),  md=2),
        dbc.Col(kpi(f'{champ_pct:.0f}%',     'Champion Revenue',    'from top-tier buyers',  'rose'),   md=2),
        dbc.Col(kpi(f'£{fut["forecast"].sum():,.0f}', '6-Mo Forecast','projected revenue',   'violet'), md=2),
    ], className='g-3', style={'marginBottom':'24px'}),

    # Row 1: Revenue trend (wide) + Sunburst
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.P('Monthly Revenue Trend', className='chart-title'),
                    html.P('Bar = monthly · Line = 3-month rolling average',
                           className='chart-subtitle'),
                ], className='chart-header'),
                dcc.Graph(id='ov-revenue', figure=fig_revenue_trend(),
                          config={'displayModeBar':True,'displaylogo':False},
                          style={'height':'340px'}),
            ], className='chart-card'),
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div([
                    html.P('Revenue by Category', className='chart-title'),
                    html.P('Click to drill into products', className='chart-subtitle'),
                ], className='chart-header'),
                dcc.Graph(id='ov-sunburst', figure=fig_category_sunburst(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'340px'}),
            ], className='chart-card'),
        ], md=4),
    ], className='g-3', style={'marginBottom':'20px'}),

    # Row 2: Channel + Products
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.P('Channel Performance', className='chart-title'),
                    html.P('Revenue and AOV by acquisition source', className='chart-subtitle'),
                ], className='chart-header'),
                dcc.Graph(id='ov-channel', figure=fig_channel_perf(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'300px'}),
            ], className='chart-card'),
        ], md=6),
        dbc.Col([
            html.Div([
                html.Div([
                    html.P('Top 12 Products', className='chart-title'),
                    html.P('Ranked by total revenue', className='chart-subtitle'),
                ], className='chart-header'),
                dcc.Graph(id='ov-products', figure=fig_top_products(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'300px'}),
            ], className='chart-card'),
        ], md=6),
    ], className='g-3'),
])

# ── Sales Trends ──────────────────────────────────────────────────────────────
page_trends = html.Div([
    page_header('Sales Trends', 'Filter by category and year — charts update live'),

    # Filter bar
    html.Div([
        html.Div([
            html.Label('Category'),
            dropdown('cat-filter',
                     [{'label':'All Categories','value':'All'}] +
                     [{'label':c,'value':c} for c in sorted(df['category'].unique())],
                     'All'),
        ], style={'minWidth':'200px','flex':'1'}),
        html.Div([
            html.Label('Year'),
            dropdown('year-filter',
                     [{'label':'Both Years','value':'All'},
                      {'label':'2024','value':'2024'},
                      {'label':'2025','value':'2025'}],
                     'All'),
        ], style={'minWidth':'160px','flex':'0 0 160px'}),
    ], className='filter-bar'),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Revenue Over Time', className='chart-title'),
                          html.P('Filtered view', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='trend-revenue', config={'displayModeBar':True,'displaylogo':False},
                          style={'height':'340px'}),
            ], className='chart-card'),
        ], md=12),
    ], className='g-3', style={'marginBottom':'20px'}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Month-on-Month Growth', className='chart-title'),
                          html.P('Green = growth · Red = decline', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='trend-mom', figure=fig_mom_growth(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'300px'}),
            ], className='chart-card'),
        ], md=6),
        dbc.Col([
            html.Div([
                html.Div([html.P('Revenue Heatmap: Day × Month', className='chart-title'),
                          html.P('Average order value per cell', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='trend-dow', figure=fig_dow_heat(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'300px'}),
            ], className='chart-card'),
        ], md=6),
    ], className='g-3'),
])

# ── Customers ─────────────────────────────────────────────────────────────────
page_customers = html.Div([
    page_header('Customer Segmentation',
                'RFM scoring (Recency · Frequency · Monetary) + K-Means clustering (k=4, silhouette-validated)'),

    html.Div([
        html.Div([
            html.Label('Highlight Segment'),
            dropdown('seg-filter',
                     [{'label':'All Segments','value':'All'}] +
                     [{'label':s,'value':s} for s in
                      ['Champions','Loyal Customers','At-Risk Customers','Lost / Inactive']],
                     'All'),
        ], style={'minWidth':'220px','flex':'0 0 220px'}),

        # Inline segment badges
        html.Div([
            html.Span('Champions',        className='badge badge-emerald', style={'marginRight':'8px'}),
            html.Span('Loyal',            className='badge badge-indigo',  style={'marginRight':'8px'}),
            html.Span('At-Risk',          className='badge badge-amber',   style={'marginRight':'8px'}),
            html.Span('Lost / Inactive',  className='badge badge-rose'),
        ], style={'display':'flex','alignItems':'flex-end','paddingBottom':'2px'}),
    ], className='filter-bar'),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('RFM Scatter: Frequency vs Lifetime Value', className='chart-title'),
                          html.P('Bubble size = monetary value · Hover for details', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='rfm-scatter', figure=fig_rfm_scatter(),
                          config={'displayModeBar':True,'displaylogo':False},
                          style={'height':'380px'}),
            ], className='chart-card'),
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div([html.P('Segment Treemap', className='chart-title'),
                          html.P('Area = total LTV · Colour = avg LTV', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='rfm-treemap', figure=fig_rfm_treemap(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'380px'}),
            ], className='chart-card'),
        ], md=4),
    ], className='g-3', style={'marginBottom':'20px'}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Segment Summary', className='chart-title')],
                         className='chart-header', style={'paddingBottom':'12px'}),
                html.Div(style={'padding':'0 8px 12px'}, children=[
                    dash_table.DataTable(
                        columns=[
                            {'name':'Segment',          'id':'segment'},
                            {'name':'Customers',        'id':'customers',
                             'type':'numeric','format':{'specifier':','}},
                            {'name':'Avg Recency (d)',  'id':'avg_recency',
                             'type':'numeric','format':{'specifier':',.0f'}},
                            {'name':'Avg Orders',       'id':'avg_frequency',
                             'type':'numeric','format':{'specifier':',.1f'}},
                            {'name':'Avg LTV (£)',      'id':'avg_monetary',
                             'type':'numeric','format':{'specifier':',.2f'}},
                            {'name':'Rev Share %',      'id':'rev_share',
                             'type':'numeric','format':{'specifier':'.1f'}},
                        ],
                        data=seg_table_data,
                        **TABLE_STYLE,
                    ),
                ]),
            ], className='chart-card'),
        ], md=12),
    ], className='g-3'),
])

# ── Retention ─────────────────────────────────────────────────────────────────
page_retention = html.Div([
    page_header('Cohort Retention Analysis',
                'Each row is a cohort (first purchase month) · Each column = months since first purchase'),

    # Stat pills
    dbc.Row([
        dbc.Col(html.Div([
            html.Div(f'{retention.mean().iloc[1]:.1f}%', className='sp-val'),
            html.Div('Month-1 retention', className='sp-lbl'),
        ], className='stat-pill'), md=3),
        dbc.Col(html.Div([
            html.Div(f'{retention.mean().iloc[3]:.1f}%', className='sp-val'),
            html.Div('Month-3 retention', className='sp-lbl'),
        ], className='stat-pill'), md=3),
        dbc.Col(html.Div([
            html.Div(f'{retention.mean().iloc[6]:.1f}%', className='sp-val'),
            html.Div('Month-6 retention', className='sp-lbl'),
        ], className='stat-pill'), md=3),
        dbc.Col(html.Div([
            html.Div(f'{retention.mean().iloc[12]:.1f}%', className='sp-val'),
            html.Div('Month-12 retention', className='sp-lbl'),
        ], className='stat-pill'), md=3),
    ], className='g-3', style={'marginBottom':'20px'}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Cohort Retention Heatmap (%)', className='chart-title'),
                          html.P('Hover each cell for exact retention', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='cohort-heatmap', figure=fig_cohort_heatmap(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'460px'}),
            ], className='chart-card'),
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div([html.P('Average Retention Curve', className='chart-title'),
                          html.P('Mean across all cohorts', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='ret-curve', figure=fig_retention_curve(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'460px'}),
            ], className='chart-card'),
        ], md=4),
    ], className='g-3'),
])

# ── Forecast ──────────────────────────────────────────────────────────────────
r2_score = mdl.score(monthly[feat], monthly['revenue'])

page_forecast = html.Div([
    page_header('Revenue Forecast',
                'OLS regression with linear trend + monthly seasonal dummies · 95% prediction intervals'),

    dbc.Row([
        dbc.Col(html.Div([
            html.Div([html.P('Actual vs Fitted vs Forecast', className='chart-title'),
                      html.P('Shaded band = 95% prediction interval', className='chart-subtitle')],
                     className='chart-header'),
            dcc.Graph(id='forecast-main', figure=fig_forecast(),
                      config={'displayModeBar':True,'displaylogo':False},
                      style={'height':'380px'}),
        ], className='chart-card'), md=12),
    ], className='g-3', style={'marginBottom':'20px'}),

    dbc.Row([
        # Forecast table
        dbc.Col([
            html.Div([
                html.Div([html.P('6-Month Forecast', className='chart-title')],
                         className='chart-header', style={'paddingBottom':'12px'}),
                html.Div(style={'padding':'0 8px 12px'}, children=[
                    dash_table.DataTable(
                        columns=[
                            {'name':'Month',    'id':'year_month_str'},
                            {'name':'Forecast', 'id':'forecast',
                             'type':'numeric','format':{'specifier':',.0f'}},
                            {'name':'Low (£)',  'id':'lower',
                             'type':'numeric','format':{'specifier':',.0f'}},
                            {'name':'High (£)', 'id':'upper',
                             'type':'numeric','format':{'specifier':',.0f'}},
                        ],
                        data=fut[['year_month_str','forecast','lower','upper']].round(0).to_dict('records'),
                        **TABLE_STYLE,
                    ),
                ]),
            ], className='chart-card', style={'height':'100%'}),
        ], md=6),

        # Model diagnostics
        dbc.Col([
            html.Div([
                html.Div([html.P('Model Diagnostics', className='chart-title')],
                         className='chart-header'),
                html.Div(style={'padding':'8px 18px 16px'}, children=[
                    *[html.Div([
                        html.Span(lbl, className='diag-label'),
                        html.Span(val, className='diag-value'),
                    ], className='diag-row')
                    for lbl, val in [
                        ('Model type',       'OLS + Seasonal Dummies'),
                        ('Training data',    'Jan 2024 – Dec 2025'),
                        ('R² (in-sample)',   f'{r2_score:.3f}'),
                        ('Residual std',     f'£{resid_std:,.0f}'),
                        ('Forecast horizon', '6 months'),
                        ('Forecast total',   f'£{fut["forecast"].sum():,.0f}'),
                        ('Confidence level', '95%'),
                    ]],
                ]),
            ], className='chart-card', style={'height':'100%'}),
        ], md=6),
    ], className='g-3'),
])

# ── Regional ──────────────────────────────────────────────────────────────────
page_regional = html.Div([
    page_header('Regional & Product Analysis', 'Revenue breakdown by UK region and best-selling products'),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Revenue by UK Region', className='chart-title'),
                          html.P('Colour gradient = revenue magnitude', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='region-bar', figure=fig_region_rev(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'340px'}),
            ], className='chart-card'),
        ], md=6),
        dbc.Col([
            html.Div([
                html.Div([html.P('Top 12 Products', className='chart-title'),
                          html.P('Ranked by total revenue', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='top-prods', figure=fig_top_products(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'340px'}),
            ], className='chart-card'),
        ], md=6),
    ], className='g-3', style={'marginBottom':'20px'}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.P('Category Breakdown', className='chart-title'),
                          html.P('Click segments to drill into products', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='cat-sunburst-r', figure=fig_category_sunburst(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'360px'}),
            ], className='chart-card'),
        ], md=5),
        dbc.Col([
            html.Div([
                html.Div([html.P('Channel Performance', className='chart-title'),
                          html.P('Revenue and average order value', className='chart-subtitle')],
                         className='chart-header'),
                dcc.Graph(id='ch-perf-r', figure=fig_channel_perf(),
                          config={'displayModeBar':False,'displaylogo':False},
                          style={'height':'360px'}),
            ], className='chart-card'),
        ], md=7),
    ], className='g-3'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 9. CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(Output('main-content','children'), Input('url','pathname'))
def render_page(path):
    return {
        '/':          page_overview,
        '/trends':    page_trends,
        '/customers': page_customers,
        '/retention': page_retention,
        '/forecast':  page_forecast,
        '/regional':  page_regional,
    }.get(path, page_overview)

@app.callback(Output('trend-revenue','figure'),
              Input('cat-filter','value'), Input('year-filter','value'))
def update_revenue(cat, yr):
    return fig_revenue_trend(cat, yr)

@app.callback(Output('rfm-scatter','figure'), Input('seg-filter','value'))
def update_rfm(seg):
    return fig_rfm_scatter(seg)

# ══════════════════════════════════════════════════════════════════════════════
# 10. RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('\n' + '='*52)
    print('  RETAIL ANALYTICS DASHBOARD')
    print(f'  Total Revenue:  £{total_rev:,.2f}')
    print(f'  YoY Growth:     {yoy:+.1f}%')
    print(f'  Customers:      {n_cust:,}')
    print(f'  Champions:      {len(champions):,} ({champ_pct:.0f}% of revenue)')
    print('='*52)
    print('\n  Open:  http://127.0.0.1:8050\n')
    app.run(debug=False, port=8050)
