import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ============= ЗАГРУЗКА ДАННЫХ =============
df = pd.read_csv('asl_video_data.csv')

print("Первые 5 строк:")
print(df.head())
print("\nПоследние 5 строк:")
print(df.tail())
print("\nИнформация о данных:")
print(df.info())
print("\nПропущенные значения:")
print(df.isna().sum())

# ============= ПОДГОТОВКА ДАННЫХ =============
X = df.drop(["label"], axis=1)  # Признаки
y = df["label"]  # Метки

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nРазмеры данных:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============= СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ =============

# 1. Логистическая регрессия
print("\nОбучение Logistic Regression...")
model_1 = LogisticRegression(max_iter=1000)
model_1.fit(X_train, y_train)

# 2. Random Forest
print("Обучение Random Forest...")
model_2 = RandomForestClassifier(n_estimators=500, random_state=42)
model_2.fit(X_train, y_train)


# ============= ФУНКЦИЯ ОЦЕНКИ =============
def score_multiclass(model, name):
    preds = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, preds)
    precision = metrics.precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = metrics.f1_score(y_test, preds, average='weighted', zero_division=0)

    print(f"\nМодель: {name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("-" * 40)

    return accuracy


# ============= ОЦЕНКА ВСЕХ МОДЕЛЕЙ =============
print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print("=" * 50)

acc1 = score_multiclass(model_1, "Logistic Regression")
acc2 = score_multiclass(model_2, "Random Forest")

pickle.dump(model_1, open('Logistic_Regression.pkl', 'wb'))
pickle.dump(model_2, open('Random_Forest.pkl', 'wb'))