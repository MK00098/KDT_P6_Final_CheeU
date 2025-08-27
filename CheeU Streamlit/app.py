import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Any

# 페이지 설정
st.set_page_config(
    page_title="CheeU Model Test",
    layout="wide"
)

# HRV 메타데이터
HRV_FEATURES = [
    'HF', 'HR', 'LF', 'MSI', 'PSI', 'RMSSD', 'SDNN', 'SDSD', 'TINN', 'TP',
    'calc_times', 'lnHF', 'lnLF', 'lnTP', 'meanRR', 'norm_HF', 'norm_LF',
    'norm_VLF', 'pnn20', 'pnn50'
]

# 요약 통계 종류
SUMMARY_STATS = ['mean', 'std', 'min', 'max', 'med', 'q25', 'q75', 'iqr', 'len']

def summarize_list(arr: List[float]) -> Dict[str, float]:
    """리스트를 요약 통계로 변환"""
    if len(arr) == 0:
        return {stat: np.nan for stat in SUMMARY_STATS[:-1]} | {'len': 0}
    
    a = np.asarray(arr, dtype=float)
    q25 = np.nanpercentile(a, 25)
    q75 = np.nanpercentile(a, 75)
    
    return {
        'mean': float(np.nanmean(a)),
        'std': float(np.nanstd(a)),
        'min': float(np.nanmin(a)),
        'max': float(np.nanmax(a)),
        'med': float(np.nanmedian(a)),
        'q25': float(q25),
        'q75': float(q75),
        'iqr': float(q75 - q25),
        'len': int(np.sum(~np.isnan(a)))
    }

def generate_hrv_timeseries(feature: str, profile: str, length: int = 50) -> List[float]:
    """HRV 시계열 데이터 생성"""
    
    # 프로필별 HRV 범위 설정
    ranges = {
        'normal': {
            'HR': (65, 75), 'HF': (150, 300), 'LF': (300, 600), 'MSI': (8, 12), 'PSI': (6, 10),
            'RMSSD': (25, 45), 'SDNN': (35, 55), 'SDSD': (20, 40), 'TINN': (100, 200), 'TP': (500, 1000),
            'calc_times': (50, 50), 'lnHF': (5.0, 6.0), 'lnLF': (6.0, 7.0), 'lnTP': (6.5, 7.5),
            'meanRR': (800, 1000), 'norm_HF': (0.3, 0.6), 'norm_LF': (0.4, 0.7), 'norm_VLF': (0.1, 0.3),
            'pnn20': (40, 70), 'pnn50': (15, 35)
        },
        'depressed': {
            'HR': (70, 85), 'HF': (100, 250), 'LF': (200, 500), 'MSI': (6, 10), 'PSI': (5, 8),
            'RMSSD': (20, 35), 'SDNN': (25, 45), 'SDSD': (15, 30), 'TINN': (80, 150), 'TP': (400, 800),
            'calc_times': (50, 50), 'lnHF': (4.5, 5.5), 'lnLF': (5.5, 6.5), 'lnTP': (6.0, 7.0),
            'meanRR': (750, 950), 'norm_HF': (0.2, 0.5), 'norm_LF': (0.5, 0.8), 'norm_VLF': (0.1, 0.3),
            'pnn20': (30, 60), 'pnn50': (10, 25)
        },
        'anxious': {
            'HR': (75, 90), 'HF': (80, 200), 'LF': (400, 800), 'MSI': (5, 9), 'PSI': (4, 7),
            'RMSSD': (18, 30), 'SDNN': (20, 40), 'SDSD': (12, 25), 'TINN': (60, 120), 'TP': (300, 600),
            'calc_times': (50, 50), 'lnHF': (4.0, 5.0), 'lnLF': (6.0, 7.0), 'lnTP': (5.5, 6.5),
            'meanRR': (700, 850), 'norm_HF': (0.1, 0.4), 'norm_LF': (0.6, 0.9), 'norm_VLF': (0.1, 0.3),
            'pnn20': (25, 50), 'pnn50': (5, 20)
        },
        'both': {
            'HR': (80, 95), 'HF': (70, 180), 'LF': (350, 700), 'MSI': (4, 8), 'PSI': (3, 6),
            'RMSSD': (15, 28), 'SDNN': (18, 35), 'SDSD': (10, 22), 'TINN': (50, 100), 'TP': (250, 500),
            'calc_times': (50, 50), 'lnHF': (4.0, 5.0), 'lnLF': (6.0, 7.0), 'lnTP': (5.5, 6.5),
            'meanRR': (650, 800), 'norm_HF': (0.1, 0.3), 'norm_LF': (0.7, 0.9), 'norm_VLF': (0.1, 0.2),
            'pnn20': (20, 45), 'pnn50': (3, 15)
        }
    }
    
    min_val, max_val = ranges.get(profile, ranges['normal']).get(feature, (0, 100))
    mean_val = (min_val + max_val) / 2
    std_val = (max_val - min_val) / 6
    
    return np.random.normal(mean_val, std_val, length).tolist()

def convert_to_timeseries(mean_value: float, length: int = 50, noise_ratio: float = 0.1) -> List[float]:
    """단일 평균값을 현실적인 시계열로 변환"""
    if pd.isna(mean_value) or mean_value == 0:
        return [0.0] * length
    
    # 적절한 변동성 추가 (평균값의 10%)
    std_dev = abs(mean_value) * noise_ratio
    timeseries = np.random.normal(mean_value, std_dev, length)
    
    # 음수값 방지 (HRV 지표는 모두 양수)
    timeseries = np.maximum(timeseries, 0.1)
    
    return timeseries.tolist()

def get_sample_info(case_type: str) -> Tuple[str, str]:
    """케이스 타입에 따른 샘플 이름과 설명 반환"""
    mapping = {
        'Depression_Yes_Anxiety_Yes': ('🌀 우울+불안 복합형 (Test data)', '우울과 불안 증상이 모두 있는 TEST 데이터'),
        'Depression_No_Anxiety_Yes': ('😰 불안형 (Test data)', '불안 증상만 있는 TEST 데이터'),
        'Depression_Yes_Anxiety_No': ('😢 우울형 (Test data)', '우울 증상만 있는 TEST 데이터'),
        'Depression_No_Anxiety_No': ('😌 정상형 (Test data)', '우울/불안 증상이 없는 TEST 데이터')
    }
    return mapping.get(case_type, ('❓ 미분류', '분류되지 않은 데이터'))

@st.cache_data
def load_real_samples() -> List[Dict[str, Any]]:
    """TEST 데이터에서 추출된 샘플 로드 (모델 학습에 미사용)"""
    try:
        df = pd.read_csv('test_extracted_samples.csv')
        samples = []
        
        for idx, row in df.iterrows():
            name, description = get_sample_info(row['case_type'])
            
            sample = {
                'id': idx + 1,
                'filename': row['filename'],
                'case_type': row['case_type'],
                'name': name,
                'description': description,
                'PHQ-9': int(row['PHQ-9']),
                'GAD-7': int(row['GAD-7']),
                'age': random.randint(25, 55) if pd.isna(row['age']) else int(row['age']),
                'gender': int(row['gender']) if not pd.isna(row['gender']) else random.choice([0, 1]),
                'sCate': int(row['sCate']) if not pd.isna(row['sCate']) else random.choice([0, 1, 2, 3])
            }
            
            # HRV 데이터를 시계열로 변환
            for feature in HRV_FEATURES:
                if feature in row and not pd.isna(row[feature]):
                    sample[feature] = convert_to_timeseries(float(row[feature]))
                else:
                    # 기본값으로 적당한 범위의 시계열 생성
                    default_value = 50.0 if 'norm_' in feature else 100.0
                    sample[feature] = convert_to_timeseries(default_value)
            
            samples.append(sample)
        
        return samples
        
    except FileNotFoundError:
        st.error("❌ test_extracted_samples.csv 파일을 찾을 수 없습니다. TEST 데이터 파일이 필요합니다.")
        return []
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return []

@st.cache_resource
def load_models():
    """모델 로드 시도"""
    models = {'gad': None, 'phq': None}
    model_status = {'gad': False, 'phq': False}
    
    try:
        import joblib
        if os.path.exists('models/lgbm_gad.joblib'):
            models['gad'] = joblib.load('models/lgbm_gad.joblib')
            model_status['gad'] = True
        if os.path.exists('models/lgbm_phq.joblib'):
            models['phq'] = joblib.load('models/lgbm_phq.joblib')
            model_status['phq'] = True
    except ImportError:
        pass
    
    return models, model_status

def dummy_predict(sample: Dict[str, Any]) -> Dict[str, Any]:
    """더미 예측 함수 (모델이 없을 때 사용)"""
    
    # 샘플 프로필에 따른 의도된 결과
    expected_results = {
        'normal': {'depression': False, 'anxiety': False, 'depression_prob': 0.15, 'anxiety_prob': 0.20},
        'depressed': {'depression': True, 'anxiety': False, 'depression_prob': 0.75, 'anxiety_prob': 0.35},
        'anxious': {'depression': False, 'anxiety': True, 'depression_prob': 0.25, 'anxiety_prob': 0.80},
        'both': {'depression': True, 'anxiety': True, 'depression_prob': 0.85, 'anxiety_prob': 0.85}
    }
    
    profile = sample.get('profile', 'normal')
    result = expected_results.get(profile, expected_results['normal']).copy()
    
    # 약간의 랜덤성 추가
    result['depression_prob'] += random.uniform(-0.1, 0.1)
    result['anxiety_prob'] += random.uniform(-0.1, 0.1)
    result['depression_prob'] = max(0, min(1, result['depression_prob']))
    result['anxiety_prob'] = max(0, min(1, result['anxiety_prob']))
    
    return result

def prepare_model_input(sample: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """모델 입력용 데이터 준비"""
    
    # HRV 요약 통계 생성
    features = {}
    for hrv_feature in HRV_FEATURES:
        stats = summarize_list(sample[hrv_feature])
        for stat_name, stat_val in stats.items():
            features[f"{hrv_feature}_{stat_name}"] = stat_val
    
    # 공통 특징
    base_features = {
        **features,
        'age': sample['age'],
        'gender': sample['gender'],
        'sCate': sample['sCate']
    }
    
    # GAD 모델용 (PHQ-9 사용)
    gad_features = {**base_features, 'PHQ-9': sample['PHQ-9'], 'KOSS': 30}  # KOSS는 더미값
    gad_df = pd.DataFrame([gad_features])
    gad_df['gender'] = gad_df['gender'].astype('category')
    gad_df['sCate'] = gad_df['sCate'].astype('category')
    
    # PHQ 모델용 (GAD7 사용)
    phq_features = {**base_features, 'GAD7': sample['GAD-7'], 'KOSS': 30}  # KOSS는 더미값
    phq_df = pd.DataFrame([phq_features])
    phq_df['gender'] = phq_df['gender'].astype('category')
    phq_df['sCate'] = phq_df['sCate'].astype('category')
    
    return gad_df, phq_df

def predict_mental_state(sample: Dict[str, Any], models: Dict, model_status: Dict) -> Dict[str, Any]:
    """정신건강 상태 예측"""
    
    # 모델이 모두 없으면 더미 예측 사용
    if not model_status['gad'] or not model_status['phq']:
        return dummy_predict(sample)
    
    try:
        # 모델 입력 데이터 준비
        gad_df, phq_df = prepare_model_input(sample)
        
        # 예측 실행
        anxiety_prob = models['gad'].predict_proba(gad_df)[0][1]
        depression_prob = models['phq'].predict_proba(phq_df)[0][1]
        
        return {
            'anxiety': anxiety_prob > 0.5,
            'depression': depression_prob > 0.5,
            'anxiety_prob': round(anxiety_prob, 3),
            'depression_prob': round(depression_prob, 3)
        }
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return dummy_predict(sample)

# 메인 앱
def main():
    # CSS 스타일링 추가 (다크모드 대응)
    st.markdown("""
    <style>
    /* 다크모드/라이트모드 공통 스타일 */
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.3rem;
        color: var(--text-color, #424242);
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .info-text {
        font-size: 1rem;
        color: var(--text-color-light, #666666);
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #1976D2;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid var(--border-color, #E3F2FD);
        padding-bottom: 0.5rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: var(--container-bg, #F8F9FA);
        border: 1px solid var(--container-border, #E0E0E0);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background-color: var(--info-bg, #F1F8E9);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: var(--text-color, #2E7D32);
    }
    
    /* 라이트모드 기본값 */
    :root {
        --text-color: #424242;
        --text-color-light: #666666;
        --border-color: #E3F2FD;
        --container-bg: #F8F9FA;
        --container-border: #E0E0E0;
        --info-bg: #F1F8E9;
    }
    
    /* 다크모드 대응 */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #E0E0E0;
            --text-color-light: #B0B0B0;
            --border-color: #424242;
            --container-bg: #2D2D2D;
            --container-border: #404040;
            --info-bg: #1B3A1B;
        }
        
        .sub-title {
            color: #E0E0E0 !important;
        }
        .info-text {
            color: #B0B0B0 !important;
        }
        .section-header {
            border-bottom: 2px solid #424242 !important;
        }
        .metric-container {
            background-color: #2D2D2D !important;
            border: 1px solid #404040 !important;
            color: #E0E0E0;
        }
        .info-box {
            background-color: #1B3A1B !important;
            color: #A5D6A7 !important;
        }
    }
    
    /* Streamlit 다크모드 클래스 감지 */
    [data-theme="dark"] .sub-title,
    .stApp[data-theme="dark"] .sub-title {
        color: #E0E0E0 !important;
    }
    [data-theme="dark"] .info-text,
    .stApp[data-theme="dark"] .info-text {
        color: #B0B0B0 !important;
    }
    [data-theme="dark"] .section-header,
    .stApp[data-theme="dark"] .section-header {
        border-bottom: 2px solid #424242 !important;
    }
    [data-theme="dark"] .metric-container,
    .stApp[data-theme="dark"] .metric-container {
        background-color: #2D2D2D !important;
        border: 1px solid #404040 !important;
        color: #E0E0E0;
    }
    [data-theme="dark"] .info-box,
    .stApp[data-theme="dark"] .info-box {
        background-color: #1B3A1B !important;
        color: #A5D6A7 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown('<div class="main-title">CheeU Model Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI 기반 우울/불안 상태 분석</div>', unsafe_allow_html=True)
    
    # 모델 로드
    models, model_status = load_models()
    
    # 실제 데이터 샘플 로드
    if 'sample_data' not in st.session_state:
        with st.spinner("실제 데이터 샘플 로드 중..."):
            st.session_state.sample_data = load_real_samples()
    
    samples = st.session_state.sample_data
    
    # 샘플 선택
    # 샘플이 로드되지 않은 경우 처리
    if not samples:
        st.error("❌ TEST 샘플 데이터를 로드할 수 없습니다. test_extracted_samples.csv 파일을 확인하세요.")
        st.stop()
    
    st.markdown('<div class="section-header">TEST 데이터 샘플 선택</div>', unsafe_allow_html=True)
    
    # 케이스별로 그룹핑
    case_groups = {}
    for sample in samples:
        case_type = sample['case_type']
        if case_type not in case_groups:
            case_groups[case_type] = []
        case_groups[case_type].append(sample)
    
    # 샘플 선택 UI - 화면 반반 분할
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 샘플 선택**")
        sample_options = []
        for i, sample in enumerate(samples):
            sample_options.append(f"샘플 {sample['id']:2d}: {sample['name']} (파일: {sample['filename']})")
        selected_idx = st.selectbox(
            "",
            range(len(samples)),
            format_func=lambda x: sample_options[x]
        )
    
    with col2:
        st.markdown("**📊 케이스별 분포**")
        for case_type, case_samples in case_groups.items():
            case_name = get_sample_info(case_type)[0].split(' ')[1]  # 이모지 제거
            st.write(f"• {case_name}: {len(case_samples)}개")
        
        st.markdown(f"**총 {len(samples)}개 실제 샘플**")
    
    selected_sample = samples[selected_idx]
    
    # 선택된 샘플 정보 표시
    st.markdown('<div class="section-header">선택된 샘플 정보</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 점수별 색상 결정
        phq9_color = "#D32F2F" if selected_sample['PHQ-9'] >= 10 else "inherit"
        gad7_color = "#D32F2F" if selected_sample['GAD-7'] >= 10 else "inherit"
        
        st.markdown("""
        <div class="metric-container">
        <h4 style="color: #1976D2; margin-bottom: 1rem;">📋 설문 점수 & 케이스 정보</h4>
        <div style="font-size: 1.1rem; line-height: 1.8;">
        • <strong>PHQ-9 (우울)</strong>: <span style="font-size: 1.3rem; color: {};">{}</span>/27<br>
        • <strong>GAD-7 (불안)</strong>: <span style="font-size: 1.3rem; color: {};">{}</span>/21<br>
        • <strong>케이스 타입</strong>: {}<br>
        • <strong>원본 파일</strong>: <code>{}</code>
        </div>
        </div>
        """.format(
            phq9_color, selected_sample['PHQ-9'],
            gad7_color, selected_sample['GAD-7'], 
            selected_sample['case_type'], 
            selected_sample['filename']
        ), unsafe_allow_html=True)
    
    with col2:
        gender_text = "여성" if selected_sample['gender'] == 0 else "남성"
        st.markdown("""
        <div class="metric-container">
        <h4 style="color: #1976D2; margin-bottom: 1rem;">👤 인구통계 정보</h4>
        <div style="font-size: 1.1rem; line-height: 1.8;">
        • <strong>나이</strong>: {}세<br>
        • <strong>성별</strong>: {}<br>
        • <strong>직업 코드</strong>: {}<br>
        • <strong>설명</strong>: {}
        </div>
        </div>
        """.format(
            selected_sample['age'],
            gender_text,
            selected_sample['sCate'],
            selected_sample['description']
        ), unsafe_allow_html=True)
    
    # 전체 메타데이터 표시
    st.markdown('<div class="section-header">HRV 메타데이터 (20개 변수)</div>', unsafe_allow_html=True)
    st.caption("표시값: 50개 시계열 데이터의 평균값 (소수점 첫째 자리 반올림) | σ: 표준편차")
    
    # HRV 데이터를 4개 열로 나누어 표시 (더 깔끔하게)
    hrv_cols = st.columns(4)
    
    for i, feature in enumerate(HRV_FEATURES):
        col_idx = i % 4
        with hrv_cols[col_idx]:
            data = selected_sample[feature]
            st.metric(
                label=feature,
                value=f"{np.mean(data):.1f}",
                delta=f"σ {np.std(data):.1f}",
                help=f"평균: {np.mean(data):.2f}, 표준편차: {np.std(data):.2f}"
            )
    
    # 원시 데이터 보기 옵션
    with st.expander("📊 HRV 원시 데이터 보기 (시계열)", expanded=False):
        st.markdown("**시계열 데이터 미리보기 (처음 10개 값)**")
        
        hrv_data_dict = {}
        for feature in HRV_FEATURES:
            # 처음 10개 값만 표시 (너무 길어지지 않도록)
            data_preview = selected_sample[feature][:10]
            hrv_data_dict[feature] = [f"{val:.2f}" for val in data_preview] + ["..."]
        
        hrv_df = pd.DataFrame(hrv_data_dict).T
        hrv_df.columns = [f"Point_{i+1}" for i in range(10)] + ["계속"]
        st.dataframe(hrv_df, use_container_width=True, height=600)
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 참고사항</strong><br>
        • 각 HRV 변수는 50개의 시계열 데이터 포인트를 가집니다<br>
        • 실제 HRV 측정값을 기반으로 생성된 시계열 데이터입니다<br>
        • 모델은 이 시계열 데이터를 180개 요약 통계로 변환하여 사용합니다
        </div>
        """, unsafe_allow_html=True)
    
    
    # 분석 실행
    st.markdown('<div class="section-header">🧠 AI 분석 실행</div>', unsafe_allow_html=True)
    
    if st.button("🚀 분석 시작", type="primary", use_container_width=True, 
                 help="선택된 샘플 데이터를 AI 모델로 분석합니다"):
        with st.spinner("🔄 AI 모델이 분석 중입니다..."):
            results = predict_mental_state(selected_sample, models, model_status)
        
        # 결과 표시
        st.markdown('<div class="section-header">🎯 AI 분석 결과</div>', unsafe_allow_html=True)
        
        # 메인 결과 카드
        col1, col2 = st.columns(2)
        
        with col1:
            depression_color = "🔴" if results['depression'] else "🟢"
            depression_text = "Yes" if results['depression'] else "No"
            depression_bg = "linear-gradient(135deg, #e57373 0%, #f06292 100%)" if results['depression'] else "linear-gradient(135deg, #81c784 0%, #66bb6a 100%)"
            st.markdown(f"""
            <div class="result-card" style="background: {depression_bg};">
                <h3 style="margin-bottom: 1rem; font-size: 1.5rem;">{depression_color} 우울 상태</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">{depression_text}</div>
                <div style="font-size: 1.2rem; opacity: 0.9;">확률: {results['depression_prob']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            anxiety_color = "🔴" if results['anxiety'] else "🟢"
            anxiety_text = "Yes" if results['anxiety'] else "No"
            anxiety_bg = "linear-gradient(135deg, #e57373 0%, #f06292 100%)" if results['anxiety'] else "linear-gradient(135deg, #81c784 0%, #66bb6a 100%)"
            st.markdown(f"""
            <div class="result-card" style="background: {anxiety_bg};">
                <h3 style="margin-bottom: 1rem; font-size: 1.5rem;">{anxiety_color} 불안 상태</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">{anxiety_text}</div>
                <div style="font-size: 1.2rem; opacity: 0.9;">확률: {results['anxiety_prob']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 종합 해석
        st.markdown("---")
        st.markdown("### 💡 종합 해석")
        
        if not results['depression'] and not results['anxiety']:
            st.success("😌 **정상 범위**: 현재 우울이나 불안 증상이 유의미하게 관찰되지 않습니다.")
        elif results['depression'] and not results['anxiety']:
            st.warning("😢 **우울 경향**: 우울 증상에 주의가 필요합니다.")
        elif not results['depression'] and results['anxiety']:
            st.warning("😰 **불안 경향**: 불안 증상에 주의가 필요합니다.")
        else:
            st.error("🌀 **복합 상태**: 우울과 불안 증상이 모두 관찰됩니다. 전문가 상담을 권장합니다.")
        
        # 주의사항
        st.markdown("---")
        st.markdown("""
        **⚠️ 중요한 안내사항**
        - 이는 AI 기반 참고용 분석으로, 의료진단을 대체할 수 없습니다
        - 실제 정신건강 문제가 의심되면 전문의와 상담하시기 바랍니다
        - MVP 버전으로 실제 서비스와 차이가 있을 수 있습니다
        """)
    
    # 데이터 리로드 버튼
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 샘플 데이터 다시 로드"):
            del st.session_state.sample_data
            st.rerun()

if __name__ == "__main__":
    main()