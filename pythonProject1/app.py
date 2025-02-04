from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
import io
import os
from torchvision import transforms
from train_cyclegan import Generator  # Импортируем напрямую

app = Flask(__name__)

# Загрузка модели
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования для изображения
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Преобразуем изображения в фиксированный размер
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
])

def load_model():
    global model
    model = Generator().to(device)  # Инициализация модели
    model.load_state_dict(torch.load("weights/G_A2B_epoch_99.pth", map_location=device, weights_only=True))
    model.eval()  # Переводим модель в режим инференса

def process_image(img):
    # Преобразуем изображение и передаем на устройство
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)  # Генерация изображения
    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()  # Переводим обратно в изображение
    output = (output * 0.5 + 0.5) * 255  # Де-нормализация
    return Image.fromarray(output.astype('uint8'))

@app.route('/')
def index():
    return render_template('index.html')  # Страница с формой загрузки

@app.route('/transform', methods=['POST'])
def transform_image():
    if 'file' not in request.files:
        return "No file uploaded", 400  # Проверяем наличие файла

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400  # Проверяем, что файл выбран

    try:
        img = Image.open(file.stream).convert('RGB')  # Открываем изображение
        result = process_image(img)  # Преобразуем изображение

        byte_io = io.BytesIO()  # Создаем объект для возврата изображения
        result.save(byte_io, 'PNG')  # Сохраняем результат в формате PNG
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')  # Отправляем преобразованное изображение
    except Exception as e:
        return f"Error: {str(e)}", 500  # Обработка ошибок

if __name__ == '__main__':
    load_model()  # Загружаем модель при старте приложения
    app.run(host='0.0.0.0', port=5000)  # Запуск приложения