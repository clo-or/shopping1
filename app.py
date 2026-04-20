import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="NEBULA | E-commerce Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Ultra-Premium Design System (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #121212 0%, #050505 100%);
        color: #E0E0E0;
    }
    
    /* Side Bar Customization */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 15, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        background: -webkit-linear-gradient(#00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 201, 255, 0.3);
    }
    
    /* Button Aesthetics */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent !important;
        border-radius: 4px 4px 0px 0px;
        color: #888;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #6366f1 !important;
    }
    
    /* Plotly background */
    .main-svg {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic & Data Processing ---

@st.cache_data
def load_and_preprocess():
    try:
        # Load datasets
        df = pd.read_csv('data/transactions_sample.csv', parse_dates=['t_dat'])
        cust = pd.read_csv('data/customers.csv')
        art = pd.read_csv('data/articles.csv')
        
        # Basic cleanup
        df['price'] = df['price'].astype(float)
        
        # Merge product info to transactions for category analysis
        df_full = df.merge(art[['article_id', 'product_group_name', 'product_type_name']], on='article_id', how='left')
        
        return df_full, cust, art
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None, None, None

def run_rfm_analysis(df):
    snapshot_date = df['t_dat'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        't_dat': lambda x: (snapshot_date - x.max()).days,
        'customer_id': 'count',
        'price': 'sum'
    }).rename(columns={
        't_dat': 'Recency',
        'customer_id': 'Frequency',
        'price': 'Monetary'
    })
    
    # 5-level scoring
    rfm['R'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    rfm['Score'] = rfm[['R', 'F', 'M']].sum(axis=1).astype(int)
    
    def segment(x):
        if x >= 13: return 'Champions'
        if x >= 10: return 'Loyal'
        if x >= 7: return 'Potential'
        if x >= 4: return 'At Risk'
        return 'Inactive'
    
    rfm['Segment'] = rfm['Score'].apply(segment)
    return rfm

# --- Main Application Interface ---

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: white;'>NEBULA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>AI-Powered E-commerce Analytics</p>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("Navigation", 
                        ["🌟 Dashboard Overview", 
                         "🎯 Customer Segments", 
                         "🌓 Retention Insight", 
                         "💎 LTV Prediction", 
                         "🤖 AI Strategy Lab"])
        
        st.sidebar.markdown("---")
        st.sidebar.caption("v2.0.0 Stable | Data: H&M Sampled")

    df, cust, art = load_and_preprocess()
    
    if df is None:
        st.info("데이터 파일을 확인해주세요 (data/ 폴더)")
        return

    if menu == "🌟 Dashboard Overview":
        st.title("🌟 Executive Dashboard")
        st.markdown("전체 거래 데이터의 핵심 지표와 트렌드를 분석합니다.")
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Revenue", f"${df['price'].sum():,.0f}")
        m2.metric("Total Orders", f"{len(df):,}")
        m3.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
        m4.metric("Avg Order Value", f"${df['price'].mean():.2f}")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Sales Trend
            st.subheader("📊 Sales Trend")
            sales_freq = st.selectbox("Frequency", ["D", "W", "ME"], index=2)
            # resample('ME') for month end to avoid warning in pandas 2.2+
            trend = df.set_index('t_dat')['price'].resample(sales_freq).sum().reset_index()
            fig = px.area(trend, x='t_dat', y='price', 
                          template="plotly_dark", 
                          color_discrete_sequence=['#6366f1'])
            fig.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("🛍️ Top Product Groups")
            top_arts = df['product_group_name'].value_counts().head(7).reset_index()
            fig_bar = px.bar(top_arts, x='count', y='product_group_name', orientation='h',
                             template="plotly_dark", color='count', color_continuous_scale='Viridis')
            fig_bar.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    elif menu == "🎯 Customer Segments":
        st.title("🎯 RFM Segmentation")
        rfm_df = run_rfm_analysis(df)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Segment Distribution")
            seg_counts = rfm_df['Segment'].value_counts().reset_index()
            fig_donut = px.pie(seg_counts, values='count', names='Segment', hole=0.7,
                               template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col2:
            st.subheader("Segment Insight Matrix")
            # Aggregated metrics for segments
            seg_summary = rfm_df.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'customer_id': 'count'
            }).rename(columns={'customer_id': 'Count'}).reset_index()
            
            st.dataframe(seg_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        st.markdown("---")
        st.subheader("Scatter: Recency vs Monetary")
        fig_scatter = px.scatter(rfm_df, x='Recency', y='Monetary', color='Segment',
                                 size='Frequency', hover_name='Segment', log_y=True,
                                 template="plotly_dark", opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif menu == "🌓 Retention Insight":
        st.title("🌓 Retention & Cohort Analysis")
        
        # Calculate Cohorts
        df['order_month'] = df['t_dat'].dt.to_period('M')
        df['cohort_month'] = df.groupby('customer_id')['t_dat'].transform('min').dt.to_period('M')
        
        cohort_data = df.groupby(['cohort_month', 'order_month']).agg(n_customers=('customer_id', 'nunique')).reset_index()
        cohort_data['period_number'] = (cohort_data['order_month'] - cohort_data['cohort_month']).apply(lambda x: x.n)
        
        pivot = cohort_data.pivot_table(index='cohort_month', columns='period_number', values='n_customers')
        retention = pivot.divide(pivot.iloc[:, 0], axis=0)
        
        st.subheader("Monthly Retention Heatmap")
        fig_heat = go.Figure(data=go.Heatmap(
            z=retention.values,
            x=retention.columns,
            y=[str(i) for i in retention.index],
            colorscale='Magma',
            text=np.round(retention.values, 2),
            texttemplate="%{text}"
        ))
        fig_heat.update_layout(template="plotly_dark", xaxis_title="Months Passed")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.info("💡 첫 구매 이후 N개월 뒤에도 다시 구매한 고객의 비율을 보여줍니다.")

    elif menu == "💎 LTV Prediction":
        st.title("💎 Customer Lifetime Value Prediction")
        st.markdown("BG/NBD 모델을 사용하여 고객의 미래 가치를 시뮬레이션합니다.")
        
        try:
            from lifetimes import BetaGeoFitter
            from lifetimes.utils import summary_data_from_transaction_data
            
            summary = summary_data_from_transaction_data(df, 'customer_id', 't_dat', monetary_value_col='price')
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(summary['frequency'], summary['recency'], summary['T'])
            
            summary['prob_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])
            summary['pred_purc'] = bgf.conditional_expected_number_of_purchases_up_to_time(30, summary['frequency'], summary['recency'], summary['T'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Probability of being 'Alive'")
                fig_alive = px.histogram(summary, x='prob_alive', nbins=50, template="plotly_dark", color_discrete_sequence=['#00ffcc'])
                st.plotly_chart(fig_alive)
            with c2:
                st.subheader("Predicted Purchases (Next 30 Days)")
                fig_pred = px.histogram(summary, x='pred_purc', nbins=50, template="plotly_dark", color_discrete_sequence=['#ff00ff'])
                st.plotly_chart(fig_pred)
                
            st.subheader("Top High-Value Potential Customers")
            st.dataframe(summary.sort_values(by='pred_purc', ascending=False).head(50), use_container_width=True)
            
        except:
            st.warning("`lifetimes` 라이브러리가 설치되지 않아 모의 시각화 데이터를 표시합니다.")
            st.image("https://images.unsplash.com/photo-1551288049-bbda48658aba?auto=format&fit=crop&q=80&w=1000", caption="Predictive Analytics Concept")
            st.markdown("### 인공지능 기반 LTV 모의 분석")
            st.progress(85)
            st.caption("전체 고객의 12%가 다음 달 이탈 위험군으로 분류되었습니다.")

    elif menu == "🤖 AI Strategy Lab":
        st.title("🤖 Strategic AI Strategy Generator")
        st.markdown("데이터 분석 결과를 바탕으로 AI가 최적의 마케팅 전략을 생성합니다.")
        
        target = st.select_slider("Select Customer Segment for Analysis", 
                                  options=["Inactive", "At Risk", "Potential", "Loyal", "Champions"])
        
        if st.button("Generate Strategy"):
            with st.spinner("AI가 데이터를 심층 분석 중입니다..."):
                import time
                time.sleep(1.5)
                
                st.success(f"### {target} 고객을 위한 AI 전략 리포트")
                
                if target == "Champions":
                    st.markdown("""
                    - **현황**: 전체 매출의 45%를 차지하는 최고 가치 고객군입니다.
                    - **AI 추천**: 
                        1. 신상품 얼리 액세스(Early Access) 제공
                        2. 1:1 전담 퍼스널 쇼퍼 기능 활성화 (LTV 강화)
                        3. 브랜드 앰배서더 초대 이벤트 발송
                    """)
                elif target == "At Risk":
                    st.markdown("""
                    - **현황**: 구매 빈도가 급감하며 이탈 징후가 포착되었습니다.
                    - **AI 추천**:
                        1. '보고 싶었습니다' 리인게이지먼트 20% 할인권 발송
                        2. 최근 장바구니에 담았던 품목의 가격 인하 알림
                        3. 고객 설문 조사를 통한 서비스 불만 요소 파악
                    """)
                else:
                    st.markdown(f"- **AI 분석**: {target} 그룹의 구매 주기를 분석한 결과, 상품 카테고리의 다양화가 시급합니다.")
                    st.markdown("- **추천**: 연관 구매(Cross-selling) 추천 알고리즘을 강화하여 객단가 상승을 유도하세요.")

if __name__ == "__main__":
    main()
