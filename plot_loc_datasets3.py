import pygmt
import pandas as pd

# 1) CSV 데이터 읽기
df = pd.read_csv('vsz_summary_3.5m_SPARSE_selected.csv')
plot_data = df[['xcoord', 'ycoord', 'Vs30_Boore2004']]

# 임시 파일 저장 (x y vs30)
temp_file = "temp_plot_data.txt"
plot_data.to_csv(temp_file, sep=' ', header=False, index=False)

# 2) 지도 영역 정의
region = [
    plot_data['xcoord'].min() - 2.5,
    plot_data['xcoord'].max() + 1.5,
    plot_data['ycoord'].min() - 1.5,
    plot_data['ycoord'].max() + 1.5
]
projection = 'M15c'

fig = pygmt.Figure()

# 3) 배경 토포맵 (고정 팔레트: dem2)
fig.grdimage(
    grid='@earth_relief_01m',
    region=region,
    projection=projection,
    shading=True,
    cmap="etopo1"
)
fig.coast(
    region=region,
    projection=projection,
    shorelines="1p",
    frame=[
        'a',
        'x+l"Longitude"',
        'y+l"Latitude"',
    ],
    
    water='white'
)


# 4) Vs30 팔레트 (디스크리트)
# 예시: 100 ~ 400 m/s 구간을 50 단위로 나눔
pygmt.makecpt(
    cmap="turbo",
    
    series=[100, 400, 20],   # [min, max, step]
    continuous=False,
    reverse=False
)

# 5) Vs30 산점도
fig.plot(
    data=temp_file,
    style="c0.25c",     # 원 크기
    cmap=True,          # colormap 적용
    pen="0.5p,black",   # 테두리
    incols="0,1,2"      # x, y, z 컬럼
)

# 6) 컬러바 추가
fig.colorbar(
    frame=["a40f20", "x+lVs30 [m/s]"],  # 눈금 50 간격, 레이블
    position="JBC+w10c/0.5c+o0/0.7c"      # 아래 중앙, 길이 10cm
)

# 7) 저장 & 출력
fig.savefig("vs30_scatter_plot_with_map_discrete.png", dpi=1200)
fig.show()

