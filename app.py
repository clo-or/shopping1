import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- Page Config ---
st.set_page_config(
    page_title="고객 RFM & LTV 분석 시스템",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #1e2130 100%);
        color: #ffffff;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background: linear-gradient(45deg, #ff4b4b, #ff7e5f);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    # In a real app, we'd handle large files more carefully
    # Using sample data if files are too large
    try:
        transactions = pd.read_csv('data/transactions_sample.csv', parse_dates=['t_dat'])
        customers = pd.read_csv('data/customers.csv')
        articles = pd.read_csv('data/articles.csv')
        return transactions, customers, articles
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None

def calculate_rfm(df):
    # Reference date (usually max date + 1)
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
    
    # RFM Scores (1-5)
    # Using rank(method='first') to handle duplicates in qcut
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # Segmentation logic
    def segment_customer(df):
        if df['RFM_Score'] >= 13:
            return 'Champions'
        elif df['RFM_Score'] >= 10:
            return 'Loyal Customers'
        elif df['RFM_Score'] >= 7:
            return 'At Risk'
        elif df['RFM_Score'] >= 4:
            return 'About to Sleep'
        else:
            return 'Lost'
            
    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

def plot_cohort_analysis(df):
    # Simplified cohort analysis (Monthly Cohorts)
    df['order_month'] = df['t_dat'].dt.to_period('M')
    df['cohort_month'] = df.groupby('customer_id')['t_dat'].transform('min').dt.to_period('M')
    
    cohort_data = df.groupby(['cohort_month', 'order_month']).agg(n_customers=('customer_id', 'nunique')).reset_index()
    cohort_data['period_number'] = (cohort_data['order_month'] - cohort_data['cohort_month']).apply(lambda x: x.n)
    
    cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='period_number', values='n_customers')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    
    # Plotting using Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=retention.values,
        x=retention.columns,
        y=[str(i) for i in retention.index],
        colorscale='Viridis',
        text=np.round(retention.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Monthly Cohort Retention Analysis", template="plotly_dark")
    return fig

# --- Main App ---
def main():
    st.sidebar.title("💎 Premium Analytics")
    menu = st.sidebar.selectbox("메뉴 선택", ["대시보드 홈", "RFM 세그먼테이션", "코호트 분석", "LTV 예측 (BG/NBD)", "ROI 시뮬레이터", "맞춤형 AI 전략"])

    transactions, customers, articles = load_data()
    
    if transactions is None:
        st.warning("데이터가 'data' 폴더에 있는지 확인해주세요.")
        return

    if menu == "대시보드 홈":
        st.title("🚀 프로젝트 대시보드")
        st.subheader("H&M 고객 데이터 기반 분석 시스템")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 거래 수", f"{len(transactions):,}")
        with col2:
            st.metric("총 고객 수", f"{transactions['customer_id'].nunique():,}")
        with col3:
            st.metric("총 매출", f"${transactions['price'].sum():.2f}")
        with col4:
            st.metric("평균 객단가", f"${transactions['price'].mean():.2f}")

        st.markdown("---")
        
        # Monthly Revenue Trend
        st.subheader("📊 월별 매출 트렌드")
        monthly_rev = transactions.set_index('t_dat')['price'].resample('ME').sum().reset_index()
        fig_rev = px.line(monthly_rev, x='t_dat', y='price', title="Monthly Revenue Trend", template="plotly_dark")
        st.plotly_chart(fig_rev, use_container_width=True)

    elif menu == "RFM 세그먼테이션":
        st.title("🎯 RFM 고객 세그먼테이션")
        rfm_df = calculate_rfm(transactions)
        
        # Treemap for segments
        st.write("### 고객 세그먼트 트리맵")
        segment_counts = rfm_df['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig_tree = px.treemap(segment_counts, path=['Segment'], values='Count', 
                             color='Count', colorscale='RdBu', template="plotly_dark")
        st.plotly_chart(fig_tree, use_container_width=True)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### 세그먼트 비율")
            fig_pie = px.pie(segment_counts, values='Count', names='Segment', hole=0.5, template="plotly_dark")
            st.plotly_chart(fig_pie)
            
        with col2:
            st.write("### 세그먼트별 지표 상세")
            seg_avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
            st.dataframe(seg_avg.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

        st.subheader("데이터 분포 (Recency vs Frequency)")
        fig_scatter = px.scatter(rfm_df.sample(1000), x='Recency', y='Frequency', color='Segment', 
                                 size='Monetary', hover_data=['customer_id'], template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif menu == "코호트 분석":
        st.title("📈 코호트 리텐션 분석")
        st.markdown("가입월 기준으로 시간이 흐름에 따라 고객이 얼마나 유지(재구매)되는지 분석합니다.")
        with st.spinner("코호트 분석 산출 중..."):
            fig_cohort = plot_cohort_analysis(transactions)
            st.plotly_chart(fig_cohort, use_container_width=True)
            st.info("💡 각 셀의 숫자는 해당 월에 가입한 고객 중 N개월 후에도 구매를 유지한 고객의 비율을 의미합니다.")

    elif menu == "LTV 예측 (BG/NBD)":
        st.title("🔮 미래 가치(LTV) 예측")
        st.info("BG/NBD 모델을 사용하여 고객의 향후 30일 내 구매 확률 및 예상 가치를 계산합니다.")
        
        # Check for lifetimes
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter
            from lifetimes.utils import summary_data_from_transaction_data
            
            # Preparation for lifetimes
            summary = summary_data_from_transaction_data(transactions, 'customer_id', 't_dat', monetary_value_col='price')
            
            # Simplified BG/NBD fitting
            bgf = BetaGeoFitter(penalizer_coef=0.0)
            bgf.fit(summary['frequency'], summary['recency'], summary['T'])
            
            summary['prob_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])
            summary['pred_num_txn'] = bgf.conditional_expected_number_of_purchases_up_to_time(30, summary['frequency'], summary['recency'], summary['T'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 고객별 이탈 위험도 (Prob Alive)")
                fig_prob = px.histogram(summary, x='prob_alive', nbins=50, title="Probability Alive Distribution", template="plotly_dark")
                st.plotly_chart(fig_prob)
            
            with col2:
                st.write("### 예상 구매 횟수 (Next 30 Days)")
                fig_pred = px.histogram(summary, x='pred_num_txn', nbins=50, title="Predicted Purchases in 30 Days", template="plotly_dark")
                st.plotly_chart(fig_pred)
                
            st.subheader("예측 데이터 상세 (상위 100명)")
            st.dataframe(summary.sort_values(by='pred_num_txn', ascending=False).head(100), use_container_width=True)
            
        except ImportError:
            st.warning("`lifetimes` 라이브러리가 설치되지 않았습니다. `pip install lifetimes`가 필요합니다.")
            st.write("모의 분석 결과:")
            st.image("https://via.placeholder.com/800x400.png?text=LTV+Analysis+Visualization+Placeholder")

    elif menu == "ROI 시뮬레이터":
        st.title("💰 마케팅 ROI 시뮬레이터")
        st.markdown("캠페인 비용과 대상 고객층을 설정하여 예상 ROI를 계산합니다.")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                target_segment = st.selectbox("타겟 세그먼트", ["Champions", "Loyal Customers", "At Risk", "About to Sleep"])
                target_count = st.number_input("대상 고객 수", value=10000)
            with col2:
                cost_per_cust = st.number_input("고객당 마케팅 비용 ($)", value=2.0)
                conversion_rate = st.slider("예상 전환율 (%)", 0.0, 100.0, 5.0)
            with col3:
                avg_profit = st.number_input("평균 기대 수익 ($)", value=50.0)
                
            total_cost = target_count * cost_per_cust
            expected_conversions = target_count * (conversion_rate / 100)
            expected_revenue = expected_conversions * avg_profit
            roi = ((expected_revenue - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            
            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("총 캠페인 비용", f"${total_cost:,.2f}")
            res_col2.metric("예상 추가 매출", f"${expected_revenue:,.2f}")
            res_col3.metric("예상 ROI", f"{roi:.1f}%", delta=f"{roi-100:.1f}%")

    elif menu == "맞춤형 AI 전략":
        st.title("🤖 AI 기반 마케팅 전략 생성")
        st.markdown("최근 트렌드와 RFM 분석 결과를 결합하여 최적의 전략을 제안합니다.")
        
        target_info = st.selectbox("전략을 생성할 세그먼트", ["At Risk", "Champions", "New Customers"])
        
        if st.button("전략 생성하기"):
            with st.spinner("AI가 데이터를 분석하여 전략을 수립 중입니다..."):
                # Mock LLM Output
                import time
                time.sleep(2)
                
                if target_info == "At Risk":
                    st.success("### At Risk 고객 대응 전략")
                    st.markdown("""
                    - **진단**: 최근 구매 주기가 길어지고 있으며 이탈 확률이 40% 이상으로 상승했습니다.
                    - **추천 액션**: 
                        1. 20% 할인 리워드 쿠폰 발송 (유효기간 7일)
                        2. '우리가 보고 싶으셨나요?' 제목의 개인화 푸시 알림
                        3. 고객이 제안했던 최근 상품 카테고리(네이버 트렌드 기반) 위주 상품 추천
                    - **기대 효과**: 이탈률 15% 감소, 재구매율 10% 상승
                    """)
                elif target_info == "Champions":
                    st.success("### Champions 고객 강화 전략")
                    st.markdown("""
                    - **진단**: 높은 구매 빈도와 가치를 유지하고 있는 핵심 고객층입니다.
                    - **추천 액션**:
                        1. VIP 전용 무료 배송 및 우선 배송 서비스 제공
                        2. 신상품 출시 전 사전 구매 기회(Early Access) 제공
                        3. 설문조사를 통한 브랜드 앰배서더 섭외
                    - **기대 효과**: 고객 생애 가치(LTV) 25% 추가 상승
                    """)
                else:
                    st.success("### 신규 고객 안착 전략")
                    st.markdown("""
                    - **진단**: 첫 구매 이후 두 번째 구매로 이어지는 단계가 중요합니다.
                    - **추천 액션**:
                        1. 첫 구매 감사 포인트 지급
                        2. 구매 상품과 연관된 코디 제안 컨텐츠 이메일 발송
                    - **기대 효과**: 리턴율 20% 상승
                    """)

if __name__ == "__main__":
    main()
