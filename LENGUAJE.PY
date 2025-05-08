import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from collections import deque

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = r"C:\Users\ACER\Documents\PROYECTO FINAL\LESSA"  # Directorio con las carpetas de cada seña
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# 1. Función para preprocesar imágenes
def preprocess_image(image):
    # Convertir a RGB (MediaPipe requiere RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detectar mano
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Obtener coordenadas de los landmarks
        h, w = image.shape[:2]
        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]
        
        # Calcular bounding box con margen
        margin = 20
        x_min, x_max = max(0, min(x_coords) - margin), min(w, max(x_coords) + margin)
        y_min, y_max = max(0, min(y_coords) - margin), min(h, max(y_coords) + margin)
        
        # Recortar región de la mano
        hand_roi = image[y_min:y_max, x_min:x_max]
        
        # Redimensionar y normalizar
        processed = cv2.resize(hand_roi, IMG_SIZE)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) / 255.0
        
        return processed
    
    return None

# 2. Cargar y preparar dataset
def load_dataset():
    classes = sorted(os.listdir(DATA_DIR))
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                processed = preprocess_image(image)
                if processed is not None:
                    images.append(processed)
                    labels.append(class_idx)
    
    # Convertir a arrays numpy
    X = np.array(images)
    y = np.array(labels)
    
    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, stratify=y_train)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes

# 3. Construir modelo
def build_model(num_classes):
    # Usaremos MobileNetV2 como base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet')
    
    # Congelar capas base
    base_model.trainable = False
    
    # Construir modelo completo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model

# 4. Entrenar modelo
def train_model(model, X_train, y_train, X_val, y_val):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('sign_language_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks)
    
    return history

# 5. Evaluar modelo
def evaluate_model(model, X_test, y_test, class_names):
    # Evaluación cuantitativa
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nPrecisión en test: {test_acc:.2%}")
    
    # Evaluación cualitativa
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Mostrar algunas predicciones
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"Real: {class_names[y_test[i]]}\nPred: {class_names[y_pred[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Diccionario simple para corrección de palabras
WORD_DICTIONARY = {
    'hola', 'mundo', 'python', 'señas', 'lenguaje', 'mano', 'letra',
    'abecedario', 'comunicación', 'aprender', 'gracias', 'por', 'favor'
}

def find_closest_word(word):
    if not word:
        return ""
    
    # Buscar coincidencias exactas primero
    if word.lower() in WORD_DICTIONARY:
        return word
    
    # Buscar palabras similares (implementación simple)
    for dict_word in WORD_DICTIONARY:
        if dict_word.startswith(word.lower()):
            return dict_word
    
    return word

# 6. Detección en tiempo real con formación de palabras
def real_time_detection(model, class_names):
    cap = cv2.VideoCapture(0)
    
    # Variables para construir la frase
    current_letters = []
    word_history = []
    last_letter = None
    last_letter_time = 0
    letter_delay = 1.0  # segundos entre letras
    space_delay = 2.0   # segundos para considerar espacio
    last_word_time = 0
    word_timeout = 3.0  # segundos para considerar nueva palabra
    
    # Buffer para suavizar las predicciones
    prediction_buffer = deque(maxlen=5)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear el frame para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Obtener timestamp actual
        current_time = time.time()
        
        # Preprocesar imagen
        processed = preprocess_image(frame)
        
        if processed is not None:
            # Predecir
            prediction = model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Solo considerar predicciones con alta confianza
            if confidence > 0.8:
                current_letter = class_names[predicted_class]
                prediction_buffer.append(current_letter)
                
                # Tomar la letra más frecuente en el buffer
                if prediction_buffer:
                    current_letter = max(set(prediction_buffer), key=prediction_buffer.count)
                
                # Lógica para agregar letras
                if current_letter != last_letter:
                    last_letter = current_letter
                    last_letter_time = current_time
                else:
                    # Si la misma letra se mantiene por el tiempo de delay
                    if current_time - last_letter_time > letter_delay:
                        # Si ha pasado suficiente tiempo para un espacio
                        if len(current_letters) > 0 and (current_time - last_letter_time) > space_delay:
                            current_letters.append(' ')
                            last_letter_time = current_time
                        elif current_letter not in current_letters[-1:]:
                            current_letters.append(current_letter)
                            last_letter_time = current_time
        
        # Convertir la lista de letras a string
        current_text = ''.join(current_letters).strip()
        
        # Auto-corrección de palabras
        if ' ' in current_text or (current_text and (current_time - last_letter_time) > word_timeout):
            words = current_text.split()
            if words:
                last_word = words[-1]
                corrected_word = find_closest_word(last_word)
                
                if corrected_word != last_word:
                    current_letters = current_letters[:-(len(last_word))] + list(corrected_word)
                    current_text = ''.join(current_letters).strip()
            
            # Agregar a historial si hay palabra completa
            if ' ' in current_text or (current_text and (current_time - last_letter_time) > word_timeout):
                completed_word = current_text.split()[-1] if current_text else ''
                if completed_word and (not word_history or completed_word != word_history[-1]):
                    word_history.append(completed_word)
                last_word_time = current_time
        
        # Mostrar resultado
        y_pos = 40
        cv2.putText(frame, 
                   f"Letra actual: {last_letter if last_letter else 'None'}",
                   (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 0), 2)
        y_pos += 40
        
        cv2.putText(frame, 
                   f"Palabra actual: {current_text}",
                   (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 255), 2)
        y_pos += 40
        
        # Mostrar sugerencia de corrección
        if current_text:
            last_word = current_text.split()[-1] if current_text else ''
            corrected_word = find_closest_word(last_word)
            if corrected_word != last_word:
                cv2.putText(frame, 
                           f"Sugerencia: {corrected_word}",
                           (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 0, 255), 2)
                y_pos += 40
        
        # Mostrar historial de palabras
        if word_history:
            cv2.putText(frame, 
                       f"Historial: {', '.join(word_history[-3:])}",
                       (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
            y_pos += 30
        
        # Dibujar landmarks
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Mostrar instrucciones
        cv2.putText(frame, 
                   "q: Salir  c: Limpiar  s: Guardar palabra",
                   (20, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        cv2.imshow('Detector de Lenguaje de Señas', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Limpiar la palabra actual
            if current_letters:
                word_history.append(''.join(current_letters).strip())
            current_letters = []
            last_letter = None
        elif key == ord('s'):
            # Guardar la palabra actual en el historial
            if current_letters:
                completed_word = ''.join(current_letters).strip()
                if completed_word:
                    word_history.append(completed_word)
                    current_letters = []
                    last_letter = None
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar el historial completo al finalizar
    if word_history:
        print("\nHistorial completo de palabras:")
        for i, word in enumerate(word_history, 1):
            print(f"{i}. {word}")

# Flujo principal
def main():
    # Cargar dataset
    print("Cargando dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = load_dataset()
    print(f"Dataset cargado. Clases: {class_names}")
    print(f"Ejemplos de entrenamiento: {len(X_train)}")
    print(f"Ejemplos de validación: {len(X_val)}")
    print(f"Ejemplos de prueba: {len(X_test)}")
    
    # Construir modelo
    print("\nConstruyendo modelo...")
    model = build_model(len(class_names))
    model.summary()
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    evaluate_model(model, X_test, y_test, class_names)
    
    # Ejecutar detección en tiempo real
    print("\nIniciando detección en tiempo real...")
    print("Instrucciones:")
    print("- Mantén la seña por 1 segundo para agregar la letra")
    print("- Mantén por más tiempo (2 segundos) para agregar espacio")
    print("- Presiona 'c' para limpiar la palabra actual")
    print("- Presiona 's' para guardar la palabra en el historial")
    print("- Presiona 'q' para salir")
    
    real_time_detection(model, class_names)

if __name__ == "__main__":
    main()