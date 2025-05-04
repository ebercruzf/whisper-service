## # Comandos principales

# Comando para levanar el servidor de Whisper
/Users/demoUser/Documents/Proyectos/IA/OpenIA/VozParaFrontDeepSeek/whisper-service/venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000


# Ruta de directorio de prueba: /Users/demoUser/Documents/Proyectos/IA/OpenIA/VozParaFrontDeepSeek/whisper-service/app

# Prueba para el endpoint desde terminal 

# curl -X POST http://localhost:11004/api/v1/transcribe -F "audio=@/Users/demoUser/Downloads/ttsMP3.com_VoiceText_2025-5-3_19-4-43.mp3"

# Whisper Transcription Service

Un servicio de API basado en FastAPI para transcribir audio a texto utilizando el modelo Whisper de OpenAI.

## Descripción

Este servicio proporciona una API RESTful que permite transcribir archivos de audio a texto utilizando el modelo Whisper de OpenAI. Soporta múltiples formatos de audio y ofrece tanto procesamiento síncrono como asíncrono.

## Características

- Transcripción de audio a texto con Whisper
- Soporte para múltiples formatos de audio (mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac)
- Procesamiento síncrono y asíncrono
- Validación de formato y tamaño de archivos
- Detección automática de idioma
- Sistema de logging detallado
- Manejo robusto de errores y timeouts
- Modo de simulación cuando Whisper no está disponible

## Requisitos

- Python 3.8+
- FFmpeg (instalado en el sistema)
- Dependencias Python listadas en `requirements.txt`

## Instalación

1. Clona este repositorio

```bash
git clone https://github.com/tu-usuario/whisper-service.git
cd whisper-service
```

2. Crea un entorno virtual e instala las dependencias

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Instala FFmpeg si aún no lo tienes instalado:

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (usando Homebrew)**:
```bash
brew install ffmpeg
```

**Windows**:
Descarga desde [ffmpeg.org](https://ffmpeg.org/download.html) y añade el directorio bin a tu PATH.

## Uso

### Iniciar el servidor

```bash
# Puerto por defecto (8000)
python -m uvicorn main:app --reload --host 0.0.0.0

# Puerto personalizado
PORT=11004 python -m uvicorn main:app --reload --host 0.0.0.0
```

### Acceso a la documentación

Una vez que el servidor está en funcionamiento, puedes acceder a la documentación interactiva de la API:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints disponibles

- `GET /`: Información general y endpoints disponibles
- `POST /transcribe`: Transcripción síncrona de audio
- `POST /transcribe-async`: Transcripción asíncrona de audio
- `GET /tasks/{task_id}`: Verificar estado de transcripción asíncrona
- `GET /health`: Verificar estado del servicio
- `GET /supported-languages`: Obtener lista de idiomas soportados

### Ejemplos de uso

#### Transcripción síncrona con curl

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "audio=@ruta/al/archivo.mp3" \
  -F "language=es"
```

Respuesta:
```json
{
  "transcription": "Texto transcrito aquí...",
  "detected_language": "es",
  "language_probability": 0.989,
  "processing_time_seconds": 3.45
}
```

#### Transcripción asíncrona con curl

```bash
curl -X POST http://localhost:8000/api/v1/transcribe-async \
  -F "audio=@ruta/al/archivo.mp3"
```

Respuesta:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

#### Verificar estado de tarea asíncrona

```bash
curl -X GET http://localhost:8000/api/v1/tasks/550e8400-e29b-41d4-a716-446655440000
```

## Configuración

El servicio puede configurarse mediante variables de entorno:

- `PORT`: Puerto en el que se ejecutará el servidor (por defecto: 8000)
- `MODEL_SIZE`: Tamaño del modelo Whisper a utilizar (por defecto: "base")
  - Opciones: "tiny", "base", "small", "medium", "large"

## Estructura del proyecto

```
whisper-service/
│
├── main.py                  # Punto de entrada principal y definición de API
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
└── temp_audio/              # Directorio para archivos temporales (creado automáticamente)
```

## Notas de desempeño

- El tamaño del modelo afecta significativamente la precisión y velocidad de transcripción:
  - Modelos más pequeños (tiny, base) son más rápidos pero menos precisos.
  - Modelos más grandes (medium, large) son más precisos pero requieren más recursos.
- El procesamiento asíncrono es recomendado para archivos de audio largos o cuando se procesan múltiples archivos.

## Solución de problemas

- Si encuentras el error `FP16 is not supported on CPU; using FP32 instead`, esto es normal cuando se ejecuta sin GPU y no afecta la funcionalidad.
- Si el modelo Whisper no se carga correctamente, el servicio intentará usar un modo de simulación para seguir funcionando.
- Verifica que FFmpeg esté correctamente instalado y disponible en el PATH del sistema.

## Licencia

[MIT](https://opensource.org/licenses/MIT)

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.
