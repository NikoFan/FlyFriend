'''
pip install opencv-python - Для работы с камерой
pip install opencv-contrib-python
pip install pioneer-sdk - для работы с дроном
pip install imutils - Для обработки изображений
pip install pyzbar
links:
Определение расстояния
https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

Книга
https://www.pyimagesearch.com/static/cv_dl_resource_guide.pdf
'''
import cv2
import numpy as np
from pyzbar import pyzbar
import sys

# === КАЛИБРОВКА ===
REAL_QR_WIDTH_CM = 5.0        # ← ЗАМЕНИТЕ на реальный размер вашего QR (в см)
KNOWN_DISTANCE_CM = 50.0      # Расстояние при калибровке (см)
REF_QR_WIDTH_PX = 100         # Ширина QR в пикселях на калибровочном фото

FOCAL_LENGTH = (REF_QR_WIDTH_PX * KNOWN_DISTANCE_CM) / REAL_QR_WIDTH_CM

QR_CONTENT = "PHONE_TRACKER"
CENTER_TOLERANCE = 80

# === ИНИЦИАЛИЗАЦИЯ КАМЕРЫ ===
print("📷 Попытка открыть камеру...")
cap = cv2.VideoCapture(0)

# Проверка открытия камеры
if not cap.isOpened():
    print("❌ ОШИБКА: Не удалось открыть камеру (индекс 0).")
    print("   Возможно, камера занята или отсутствует.")
    sys.exit(1)

# Пробуем получить первый кадр
print("📸 Получение первого кадра для определения разрешения...")
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("❌ ОШИБКА: Камера открыта, но кадр не получен.")
    cap.release()
    sys.exit(1)

FRAME_H, FRAME_W = test_frame.shape[:2]
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2
print(f"✅ Камера готова. Разрешение: {FRAME_W}x{FRAME_H}")

# === ОСНОВНОЙ ЦИКЛ ===
try:
    while True:
        # Получение изображения с камеры
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️  Пропущен кадр. Продолжаю...")
            continue

        # Рисуем сетку зон
        cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_H), (255, 255, 255), 1)
        cv2.line(frame, (0, CENTER_Y), (FRAME_W, CENTER_Y), (255, 255, 255), 1)

        # Декодируем QR
        decoded = pyzbar.decode(frame)
        found = False

        for obj in decoded:
            try:
                data = obj.data.decode("utf-8")
                if data == QR_CONTENT:
                    found = True
                    x, y, w, h = obj.rect
                    cx, cy = x + w // 2, y + h // 2

                    # Определяем зону
                    if abs(cx - CENTER_X) < CENTER_TOLERANCE and abs(cy - CENTER_Y) < CENTER_TOLERANCE:
                        zone_text = "Center"
                    else:
                        dx = 1 if cx > CENTER_X else -1
                        dy = 1 if cy < CENTER_Y else -1
                        zone_map = {(1,1): "I", (-1,1): "II", (-1,-1): "III", (1,-1): "IV"}
                        zone_text = f"Zone {zone_map.get((dx, dy), '?')}"

                    # Расстояние
                    dist = (REAL_QR_WIDTH_CM * FOCAL_LENGTH) / w if w > 0 else 0

                    # Визуализация
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"Range: {dist:.1f} cm", (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, zone_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    break
            except Exception as e:
                print(f"⚠️  Ошибка при обработке QR: {e}")
                continue

        if not found:
            cv2.putText(frame, "Show QR-code", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображаем кадр
        cv2.imshow("Phone Tracker", frame)

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Прервано пользователем")

finally:
    print("🧹 Освобождение ресурсов...")
    cap.release()
    cv2.destroyAllWindows()