import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터 불러오기
data = pd.read_csv(r'C:\Users\user\PycharmProjects\pythonProject\CP2\data\countries of the world.csv')

# 데이터 확인
print(data.head())

# 데이터 전처리
data = data.dropna()  # 결측치 제거 등 다양한 전처리 수행

# 시각화를 통한 데이터 탐색
sns.histplot(data['Population'], bins=20)
plt.title('Distribution of Population')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.show()

# 데이터 준비
X = data.drop(['Country', 'Region', 'Population'], axis=1)  # 독립 변수 설정
y = data['Deathrate']  # 종속 변수 설정

# 학습용 데이터와 테스트용 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 머신러닝 모델 학습 (랜덤 포레스트 분류기 사용 예시)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 모델 예측
predictions = model.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# 모델 성능 평가 (혼동 행렬)
conf_matrix = confusion_matrix(y_test, predictions)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
