# 🚀 Streamlit Cloud 배포 가이드

## 📋 배포 준비사항

### 1. 파일 구조 확인
```
practical_project/
├── streamlit_app_deploy.py    # 배포용 메인 앱
├── requirements.txt           # 패키지 의존성
├── .streamlit/
│   └── config.toml           # Streamlit 설정
└── csv/
    └── 스마트팜_수정데이터.csv  # 데이터 파일
```

### 2. GitHub에 푸시
```bash
git add .
git commit -m "Add Streamlit deployment files"
git push origin main
```

### 3. Streamlit Cloud 배포

1. **Streamlit Cloud 접속**: https://share.streamlit.io/
2. **GitHub 계정 연결**
3. **Repository 선택**: `jiwon512/practical_project`
4. **Main file path**: `streamlit_app_deploy.py`
5. **Deploy!**

## 🔧 배포 시 주의사항

### 1. 메모리 제한
- Streamlit Cloud는 메모리 제한이 있음
- 모델 학습을 위해 `n_estimators`를 100으로 줄임
- CV를 3-fold로 줄임

### 2. 데이터 경로
- 상대 경로 사용: `csv/스마트팜_수정데이터.csv`
- 절대 경로는 배포 환경에서 작동하지 않음

### 3. 패키지 버전
- `>=` 사용으로 호환성 확보
- 최소 버전 요구사항 명시

## 🐛 문제 해결

### 1. 패키지 설치 오류
```bash
# requirements.txt에서 버전 충돌 확인
pip install -r requirements.txt --upgrade
```

### 2. 메모리 부족
- 모델 파라미터 줄이기
- 데이터 샘플링
- 캐싱 활용

### 3. 데이터 로드 오류
- 파일 경로 확인
- 인코딩 설정 확인
- 파일 크기 확인

## 📊 배포 후 확인사항

1. **앱 로딩**: 첫 로딩 시 모델 학습 시간 확인
2. **기능 테스트**: 농장별/개체별 예측 기능 확인
3. **성능 모니터링**: 메모리 사용량 및 응답 시간 확인

## 🔄 업데이트 방법

1. 로컬에서 코드 수정
2. GitHub에 푸시
3. Streamlit Cloud에서 자동 재배포

## 📞 지원

배포 문제 발생 시:
1. Streamlit Cloud 로그 확인
2. GitHub Issues 등록
3. Streamlit Community 포럼 활용 