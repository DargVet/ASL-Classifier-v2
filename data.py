import cv2
import csv
import mediapipe as mp

# ============= НАСТРОЙКА КАМЕРЫ =============
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # ширина
cap.set(4, 720)  # высота

# ============= НАСТРОЙКА MEDIAPIPE =============
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Детектор рук (только одна рука, уверенность 70%)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ============= СОЗДАНИЕ ФАЙЛА ДАННЫХ =============
# Создаем CSV файл и записываем заголовки
with open('asl_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Заголовки: 21 точка × 3 координаты (x, y, z) = 63 столбца + метка
    headers = []
    for i in range(21):
        headers.append(f'p{i}_x')
        headers.append(f'p{i}_y')
        headers.append(f'p{i}_z')
    headers.append('label')

    writer.writerow(headers)

# Текущая буква для сбора
current_letter = 'A'
print(f"Начинаем сбор данных для буквы: {current_letter}")

# ============= ОСНОВНОЙ ЦИКЛ =============
while True:
    # Чтение кадра с камеры
    success, img = cap.read()
    if not success:
        print("Ошибка чтения кадра")
        break

    # Конвертация в RGB для MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обработка руки
    result = hands.process(img_rgb)

    # Если рука найдена - рисуем точки и соединения
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    # Показываем текущую букву на экране
    cv2.putText(img, f'Буква: {current_letter}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение окна
    cv2.imshow('Сбор данных ASL', img)

    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF

    # Выход по 'q'
    if key == ord('q'):
        break

    # Сохранение данных по пробелу
    elif key == ord(' '):
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            wrist = hand.landmark[0]  # точка запястья (0 индекс)

            # Нормализация координат относительно запястья
            row = []
            for point in hand.landmark:
                row.append(point.x - wrist.x)  # относительный x
                row.append(point.y - wrist.y)  # относительный y
                row.append(point.z - wrist.z)  # относительный z

            row.append(current_letter)  # метка класса

            # Запись в CSV
            with open('asl_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Сохранен пример для буквы: {current_letter}")
        else:
            print("Рука не найдена!")

    # Смена буквы по '/'
    elif key == ord('/'):
        # Переключаем буквы A-Z
        current_letter = chr((ord(current_letter) - ord('A') + 1) % 26 + ord('A'))
        print(f"Переключили на букву: {current_letter}")

# ============= ОЧИСТКА =============
cap.release()
cv2.destroyAllWindows()
print("Сбор данных завершен!")