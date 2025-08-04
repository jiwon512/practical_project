import pandas as pd
import requests
import json
from datetime import datetime
import time

def get_weather_data(lat, lng, timestamp, api_key):
    """
    특정 위치와 시간의 날씨 데이터를 가져오는 함수
    
    Args:
        lat (float): 위도
        lng (float): 경도
        timestamp (datetime): 시간
        api_key (str): OpenWeatherMap API 키
    
    Returns:
        dict: 날씨 데이터 (온도, 습도, 풍속 등)
    """
    # Unix timestamp로 변환
    unix_timestamp = int(timestamp.timestamp())
    
    # OpenWeatherMap API 호출 (과거 날씨 데이터)
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        'lat': lat,
        'lon': lng,
        'dt': unix_timestamp,
        'appid': api_key,
        'units': 'metric'  # 섭씨 온도 사용
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            weather = data['data'][0]
            return {
                'temperature': weather.get('temp', None),  # 온도 (섭씨)
                'humidity': weather.get('humidity', None),  # 습도 (%)
                'wind_speed': weather.get('wind_speed', None),  # 풍속 (m/s)
                'wind_direction': weather.get('wind_deg', None),  # 풍향 (도)
                'pressure': weather.get('pressure', None),  # 기압 (hPa)
                'weather_description': weather.get('weather', [{}])[0].get('description', None)
            }
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return None
    except KeyError as e:
        print(f"데이터 파싱 오류: {e}")
        return None

def add_weather_to_dataframe(df, location_df, api_key, sample_size=None):
    """
    데이터프레임에 날씨 정보를 추가하는 함수
    
    Args:
        df (DataFrame): 스마트팜 데이터
        location_df (DataFrame): 위치 정보 (lat, lng 포함)
        api_key (str): OpenWeatherMap API 키
        sample_size (int): 테스트용 샘플 크기 (None이면 전체 데이터)
    
    Returns:
        DataFrame: 날씨 정보가 추가된 데이터프레임
    """
    # 농장아이디와 위치 정보 매핑
    location_mapping = location_df.set_index('name')[['lat', 'lng']].to_dict('index')
    
    # 샘플 데이터 사용 (API 호출 제한 때문)
    if sample_size:
        df_sample = df.head(sample_size).copy()
    else:
        df_sample = df.copy()
    
    # 날씨 정보 컬럼 초기화
    weather_columns = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'weather_description']
    for col in weather_columns:
        df_sample[f'weather_{col}'] = None
    
    print(f"총 {len(df_sample)}개 데이터에 대해 날씨 정보를 가져오는 중...")
    
    for idx, row in df_sample.iterrows():
        farm_id = row['농장아이디']
        
        # 농장 위치 정보 가져오기
        if farm_id in location_mapping:
            lat = location_mapping[farm_id]['lat']
            lng = location_mapping[farm_id]['lng']
            
            # 착유시작일시 사용
            timestamp = pd.to_datetime(row['착유시작일시'])
            
            # 날씨 데이터 가져오기
            weather_data = get_weather_data(lat, lng, timestamp, api_key)
            
            if weather_data:
                for col in weather_columns:
                    df_sample.at[idx, f'weather_{col}'] = weather_data[col]
            
            # API 호출 제한을 위한 대기 (분당 60회 제한)
            time.sleep(1)
        
        # 진행상황 출력
        if (idx + 1) % 10 == 0:
            print(f"진행률: {idx + 1}/{len(df_sample)}")
    
    return df_sample

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv("/Users/Jiwon/Documents/GitHub/practical_project/csv/스마트팜_수정데이터.csv", encoding="cp949")
    
    # 위치 데이터 로드 (shapefile에서)
    import geopandas as gpd
    location_gdf = gpd.read_file("/Users/Jiwon/Documents/GitHub/practical_project/csv/point_1.shp")
    location_df = location_gdf[['name', 'lat', 'lng']].copy()
    location_df['name'] = location_df['name'].astype(int)
    
    print("위치 데이터:")
    print(location_df)
    
    print("\n스마트팜 데이터 정보:")
    print(df.info())
    
    # API 키 설정 (실제 사용시에는 환경변수나 설정 파일에서 가져오세요)
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # 실제 API 키로 교체 필요
    
    # 날씨 정보 추가 (테스트용으로 10개 샘플만)
    df_with_weather = add_weather_to_dataframe(df, location_df, API_KEY, sample_size=10)
    
    print("\n날씨 정보가 추가된 데이터:")
    print(df_with_weather[['농장아이디', '착유시작일시', 'weather_temperature', 'weather_humidity', 'weather_wind_speed']].head())
    
    # 결과 저장
    # df_with_weather.to_csv("/Users/Jiwon/Documents/GitHub/practical_project/csv/스마트팜_날씨추가.csv", encoding="cp949", index=False) 