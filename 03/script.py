from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Создаем Flask app
app = Flask(__name__)

# Загрузка модели
try:
    model_data = joblib.load('iris_model_random_forest.pkl')
    
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        target_names = model_data.get('target_names', ['setosa', 'versicolor', 'virginica'])
        feature_names = model_data.get('feature_names', ['Длина чашелистика', 'Ширина чашелистика', 'Длина лепестка', 'Ширина лепестка'])
    else:
        model = model_data
        target_names = ['setosa', 'versicolor', 'virginica']
        feature_names = ['Длина чашелистика', 'Ширина чашелистика', 'Длина лепестка', 'Ширина лепестка']
    
    print("✅ Модель успешно загружена!")
    
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None
    target_names = ['setosa', 'versicolor', 'virginica']
    feature_names = ['Длина чашелистика', 'Ширина чашелистика', 'Длина лепестка', 'Ширина лепестка']

@app.route('/')
def index():
    """Главная страница с формой ввода"""
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Обработка предсказания"""
    if model is None:
        return render_template('result.html', error="Модель не загружена.")

    try:
        # Получаем данные из формы
        features = []
        for feature in feature_names:
            value = request.form.get(feature, '').strip()
            if not value:
                return render_template('result.html', error=f"Заполните поле '{feature}'")
            try:
                features.append(float(value))
            except ValueError:
                return render_template('result.html', error=f"Некорректное значение в поле '{feature}'")

        # Предсказание
        features_array = np.array([features])
        predicted_class = model.predict(features_array)[0]
        class_name = target_names[predicted_class]
        
        # Вероятности
        probabilities = model.predict_proba(features_array)[0]
        prob_percentages = [round(prob * 100, 2) for prob in probabilities]
        
        class_probabilities = []
        for i, prob in enumerate(prob_percentages):
            class_probabilities.append({
                'class_name': target_names[i],
                'probability': prob,
                'is_max': prob == max(prob_percentages)
            })
        
        # Русские названия
        class_translations = {
            'setosa': 'Сетоса (Ирис щетинистый)',
            'versicolor': 'Версиколор (Ирис разноцветный)',
            'virginica': 'Виргиника (Ирис виргинский)'
        }
        
        russian_class_name = class_translations.get(class_name.lower(), class_name)
        
        return render_template('result.html',
                             features=features,
                             feature_names=feature_names,
                             class_name=russian_class_name,
                             class_probabilities=class_probabilities,
                             confidence=max(prob_percentages))
    
    except Exception as e:
        return render_template('result.html', error=f"Ошибка: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)