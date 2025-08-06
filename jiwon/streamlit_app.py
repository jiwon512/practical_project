import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 페이지 설정
st.set_page_config(
    page_title="스마트팜 착유량 예측 시스템",
    page_icon="🐄",
    layout="wide"
)

# 제목
st.title("🐄 스마트팜 착유량 예측 시스템")
st.markdown("---")

@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        # 데이터 로드 (로컬 테스트용 경로)
        df = pd.read_csv("../csv/스마트팜_수정데이터.csv", encoding="cp949")
        
        # 날짜 컬럼 변환
        date_cols = ['착유시작일시', '착유종료일시']
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        
        # 착유시간 계산
        df['착유시간'] = (df['착유종료일시'] - df['착유시작일시']).dt.total_seconds() / 60
        
        # 착유시간대 구분
        df['착유시간대'] = df['착유시작일시'].dt.hour
        df['착유시간대'] = pd.cut(
            df['착유시간대'],
            bins=[0,6,12,18,24],
            right=False,
            labels=[1,2,3,4]
        ).cat.codes + 1
        
        # 파생변수 생성
        df["P/F_ratio"] = df["유단백"] / df["유지방"]
        df["착유효율"] = df["착유량"] / df["착유시간"]
        df["개체별_전도도율"] = df.groupby("개체번호")["전도도"].transform("mean")
        df['공기흐름_비율'] = df['공기흐름'] / df['착유시간']
        
        # 측정일 추출 (숫자로 변환)
        df['측정일'] = df['착유시작일시'].dt.strftime('%Y%m%d').astype(int)
        
        # 개체별 착유일수
        df['개체별_착유일수'] = df.groupby('개체번호')['측정일'].transform('nunique')
        
        # 개체별 일별 착유횟수
        df['개체별_일별착유횟수'] = df.groupby(['개체번호','측정일'])['착유회차'].transform('count')
        
        # 나이 계산 (개체번호에서 추출)
        df['나이'] = df['개체번호'].astype(str).str[4:6].astype(int)
        
        # 컬럼명 정리
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        
        # 이상치 제거
        outlier_cols = ['착유량', '착유회차', '전도도', '온도', '유지방', '유단백', '공기흐름', '착유시간']
        bounds = {}
        for col in outlier_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            bounds[col] = (lower, upper)
        
        mask = pd.Series(True, index=df.index)
        for col, (lower, upper) in bounds.items():
            mask &= df[col].between(lower, upper)
        
        df_clean = df.loc[mask].copy()
        
        return df_clean
        
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

@st.cache_resource
def train_model(df_clean):
    """모델 학습"""
    try:
        # 타겟과 피처 분리
        y = df_clean['착유량']
        X = df_clean[[
            "개체번호","공기흐름_비율","P/F_ratio","개체별_착유일수",
            "측정일","농장아이디","착유시간","착유회차",
            "착유시간대","온도","나이","혈액흐름",
            "개체별_일별착유횟수","유지방","유단백"
        ]]
        
        # 범주형 인코딩 (혈액흐름도 포함)
        cat_feats = ['농장아이디','개체번호','혈액흐름']
        te = TargetEncoder(cols=cat_feats)
        X_enc = te.fit_transform(X, y)
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.2, random_state=42
        )
        
        # 개별 모델 정의
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        lgb = LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        
        # Voting 앙상블
        voting = VotingRegressor([('rf', rf), ('xgb', xgb), ('lgb', lgb)])
        voting.fit(X_train, y_train)
        
        # Stacking 앙상블
        stack = StackingRegressor(
            estimators=[('rf', rf), ('xgb', xgb), ('lgb', lgb)],
            final_estimator=Ridge(),
            cv=5,
            n_jobs=-1
        )
        stack.fit(X_train, y_train)
        
        # 파이프라인 생성
        pipeline = Pipeline([
            ('te', te),
            ('stack', stack)
        ])
        
        return pipeline, X_train.columns.tolist()
        
    except Exception as e:
        st.error(f"모델 학습 오류: {e}")
        return None, None

def predict_farm_yield(pipeline, farm_id, date, feature_columns):
    """농장별 착유량 예측"""
    try:
        # 날짜를 숫자 형식으로 변환 (YYYYMMDD)
        date_int = int(date.replace('-', ''))
        
        # 기본값으로 예측 데이터 생성
        sample_data = pd.DataFrame({
            '개체번호': [20278],  # 기본 개체번호
            '공기흐름_비율': [0.5],
            'P/F_ratio': [3.0],
            '개체별_착유일수': [30],
            '측정일': [date_int],
            '농장아이디': [farm_id],
            '착유시간': [10],
            '착유회차': [1],
            '착유시간대': [1],
            '온도': [39.0],
            '나이': [8],
            '혈액흐름': ['N'],  # TargetEncoder로 인코딩됨
            '개체별_일별착유횟수': [2],
            '유지방': [3.5],
            '유단백': [3.2]
        })
        
        # 예측
        prediction = pipeline.predict(sample_data)[0]
        return prediction
        
    except Exception as e:
        st.error(f"예측 오류: {e}")
        return None

def predict_individual_yield(pipeline, individual_data):
    """개체별 착유량 예측"""
    try:
        prediction = pipeline.predict(individual_data)[0]
        return prediction
    except Exception as e:
        st.error(f"예측 오류: {e}")
        return None

def main():
    # 데이터 로드
    with st.spinner("데이터를 로드하는 중..."):
        df_clean = load_data()
    
    if df_clean is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 모델 학습
    with st.spinner("모델을 학습하는 중..."):
        pipeline, feature_columns = train_model(df_clean)
    
    if pipeline is None:
        st.error("모델을 학습할 수 없습니다.")
        return
    
    # 사이드바
    st.sidebar.title("📊 예측 옵션")
    prediction_type = st.sidebar.selectbox(
        "예측 유형 선택",
        ["농장별 예측", "개체별 예측"]
    )
    
    if prediction_type == "농장별 예측":
        st.header("🏭 농장별 착유량 예측")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 농장 선택
            farm_ids = sorted(df_clean['농장아이디'].unique())
            selected_farm = st.selectbox("농장 선택", farm_ids)
            
            # 예측 기간 선택
            prediction_period = st.selectbox(
                "예측 기간",
                ["다음주", "다음달"]
            )
        
        with col2:
            # 예측 날짜 설정
            today = datetime.now()
            if prediction_period == "다음주":
                target_date = today + timedelta(days=7)
            else:
                target_date = today + timedelta(days=30)
            
            st.write(f"**예측 날짜:** {target_date.strftime('%Y-%m-%d')}")
        
        # 예측 실행
        if st.button("🚀 예측 실행", type="primary"):
            with st.spinner("예측 중..."):
                prediction = predict_farm_yield(
                    pipeline, selected_farm, 
                    target_date.strftime('%Y-%m-%d'), 
                    feature_columns
                )
                
                if prediction is not None:
                    # 결과 표시
                    st.success("✅ 예측 완료!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="예측 착유량",
                            value=f"{prediction:.1f}L",
                            delta=f"{prediction - 12:.1f}L"  # 평균 대비
                        )
                    
                    with col2:
                        st.metric(
                            label="예측 기간",
                            value=prediction_period
                        )
                    
                    with col3:
                        st.metric(
                            label="농장 ID",
                            value=selected_farm
                        )
                    
                    # 농장별 통계 정보
                    farm_stats = df_clean[df_clean['농장아이디'] == selected_farm]
                    
                    if not farm_stats.empty:
                        st.subheader("📈 농장 통계 정보")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("평균 착유량", f"{farm_stats['착유량'].mean():.1f}L")
                        with col2:
                            st.metric("개체 수", len(farm_stats['개체번호'].unique()))
                        with col3:
                            st.metric("총 착유 횟수", len(farm_stats))
                        with col4:
                            st.metric("평균 착유시간", f"{farm_stats['착유시간'].mean():.1f}분")
                        
                        # 착유량 분포 차트
                        fig = px.histogram(
                            farm_stats, 
                            x='착유량', 
                            nbins=20,
                            title=f"농장 {selected_farm} 착유량 분포"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    else:  # 개체별 예측
        st.header("🐄 개체별 착유량 예측")
        
        # 개체 선택
        farm_ids = sorted(df_clean['농장아이디'].unique())
        selected_farm = st.selectbox("농장 선택", farm_ids)
        
        # 선택된 농장의 개체들
        farm_individuals = df_clean[df_clean['농장아이디'] == selected_farm]['개체번호'].unique()
        selected_individual = st.selectbox("개체 선택", sorted(farm_individuals))
        
        # 개체 정보 표시
        individual_data = df_clean[df_clean['개체번호'] == selected_individual]
        
        if not individual_data.empty:
            st.subheader("📋 개체 정보")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균 착유량", f"{individual_data['착유량'].mean():.1f}L")
            with col2:
                st.metric("총 착유 횟수", len(individual_data))
            with col3:
                st.metric("평균 착유시간", f"{individual_data['착유시간'].mean():.1f}분")
            with col4:
                st.metric("나이", f"{individual_data['나이'].iloc[0]}세")
            
            # 개체별 착유량 추이
            fig = px.line(
                individual_data.sort_values('착유시작일시'),
                x='착유시작일시',
                y='착유량',
                title=f"개체 {selected_individual} 착유량 추이"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 예측 실행
            if st.button("🚀 개체별 예측 실행", type="primary"):
                with st.spinner("예측 중..."):
                    # 최근 데이터로 예측
                    recent_data = individual_data.iloc[-1:].copy()
                    prediction = predict_individual_yield(pipeline, recent_data)
                    
                    if prediction is not None:
                        st.success("✅ 예측 완료!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="다음 착유 예측량",
                                value=f"{prediction:.1f}L",
                                delta=f"{prediction - individual_data['착유량'].mean():.1f}L"
                            )
                        
                        with col2:
                            st.metric(
                                label="개체 번호",
                                value=selected_individual
                            )
    
    # 모델 정보
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 모델 정보")
    st.sidebar.info("""
    **스택킹 앙상블 모델**
    - Random Forest
    - XGBoost  
    - LightGBM
    - 메타모델: Ridge
    """)
    
    # 데이터 정보
    st.sidebar.subheader("📊 데이터 정보")
    st.sidebar.info(f"""
    **총 데이터**: {len(df_clean):,}개
    **농장 수**: {len(df_clean['농장아이디'].unique())}개
    **개체 수**: {len(df_clean['개체번호'].unique())}개
    **기간**: {df_clean['착유시작일시'].min().strftime('%Y-%m-%d')} ~ {df_clean['착유시작일시'].max().strftime('%Y-%m-%d')}
    """)

if __name__ == "__main__":
    main() 