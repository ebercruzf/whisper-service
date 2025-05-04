from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import logging
import uuid
import shutil
import signal
from typing import Dict, Optional, List, Union, Any
import importlib.util
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Configurar logging con más detalle
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("whisper_service.log")
    ]
)
logger = logging.getLogger("whisper-service")

# Función para verificar dependencias
#def check_dependency(module_name: str) -> bool:
#    """Verifica si un módulo Python está instalado"""
#    return importlib.util.find_spec(module_name) is not None



# Verificar dependencias críticas
#required_dependencies = {
#    "fastapi": "pip install fastapi",
#    "multipart": "pip install python-multipart",
    #"multipart": "pip install python-multipart==0.0.20",
#}
required_dependencies = {
    "fastapi": "pip install fastapi",
    "multipart": "pip install python-multipart",  # <-- ¡Clave correcta!
}

def check_dependency(module_name: str) -> bool:
    """Verifica si un módulo Python está instalado usando importación real"""
    try:
        __import__(module_name)
        return True
    except Exception as e:  # Captura cualquier error durante la importación
        logger.warning(f"Error al importar {module_name}: {str(e)}")
        return False
        
missing_dependencies = []
for dep, install_cmd in required_dependencies.items():
    if not check_dependency(dep):
        missing_dependencies.append(f"{dep} (instalar con: {install_cmd})")

if missing_dependencies:
    logger.error(f"Faltan dependencias requeridas: {', '.join(missing_dependencies)}")
    print(f"ERROR: Faltan dependencias requeridas: {', '.join(missing_dependencies)}")
    sys.exit(1)

# Importar whisper solo si está disponible
whisper = None
try:
    import whisper
    logger.info("Módulo whisper importado correctamente")
except ImportError:
    logger.warning("Módulo whisper no encontrado. Utilizando modo de simulación.")
    # Crear un simulador de Whisper si no está disponible
    class MockWhisperModel:
        def __init__(self, name="base"):
            self.model_name = name
            
        def transcribe(self, audio_path, **kwargs):
            # Simula transcripción para pruebas
            language = kwargs.get("language", "es")
            return {
                "text": f"Esta es una transcripción simulada del archivo {audio_path}. Whisper no está disponible en este momento.",
                "language": language,
                "language_probability": 0.98
            }
    
    # Módulo falso de whisper
    class whisper:
        @staticmethod
        def load_model(model_name):
            return MockWhisperModel(model_name)

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Whisper Transcription Service",
    description="API para transcribir audio a texto usando Whisper",
    version="1.0.0",
)

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe a tus dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache para transcripciones en curso
transcription_tasks: Dict[str, Dict] = {}

# Variable para almacenar el modelo (lazy loading)
_model = None

# Implementar un timeout para evitar bloqueos
class timeout:
    def __init__(self, seconds=30, error_message='Timeout excedido'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        # Solo establecer la alarma en sistemas que soporten SIGALRM (Unix/Linux/Mac)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, type, value, traceback):
        # Cancelar la alarma solo si el sistema lo soporta
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

def get_model():
    """Carga el modelo Whisper de forma perezosa con manejo de errores"""
    global _model
    if _model is None:
        try:
            logger.info("Loading Whisper model...")
            model_name = os.getenv("MODEL_SIZE", "base")
            with timeout(seconds=60, error_message="Timeout al cargar el modelo Whisper"):
                _model = whisper.load_model(model_name)
            logger.info(f"Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            # Crear un modelo simulado para permitir que el servicio siga funcionando
            if whisper:
                _model = MockWhisperModel("base")
            else:
                raise RuntimeError(f"No se pudo cargar el modelo Whisper: {str(e)}")
    return _model

# Dependencia para validar formatos de archivo de audio
def validate_audio_file(file: UploadFile = File(...)) -> UploadFile:
    """Valida que el archivo sea un formato de audio soportado"""
    valid_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg', '.flac']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if not any(file_ext.endswith(ext) for ext in valid_extensions):
        logger.warning(f"Archivo con formato no soportado: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de archivo no soportado. Formatos válidos: {', '.join(valid_extensions)}"
        )
    
    # Validar tamaño máximo (50MB)
    max_size = 50 * 1024 * 1024  # 50MB en bytes
    if hasattr(file, 'size') and file.size > max_size:
        logger.warning(f"Archivo demasiado grande: {file.size} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"El archivo es demasiado grande. Tamaño máximo: 50MB"
        )
        
    return file

@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "Bienvenido al servicio de transcripción con Whisper",
        "docs": "/docs",
        "endpoints": [
            {"path": "/transcribe", "method": "POST", "description": "Transcribir audio de forma síncrona"},
            {"path": "/transcribe-async", "method": "POST", "description": "Transcribir audio de forma asíncrona"},
            {"path": "/tasks/{task_id}", "method": "GET", "description": "Verificar estado de una tarea asíncrona"},
            {"path": "/health", "method": "GET", "description": "Verificar estado del servicio"}
        ]
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = Depends(validate_audio_file),
    language: Optional[str] = Form(None)
):
    """
    Transcribe un archivo de audio usando Whisper.
    
    - **file**: Archivo de audio a transcribir (formatos soportados: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac)
    - **language**: Código de idioma opcional (ej: "es" para español). Si no se especifica, se detectará automáticamente.
    
    Retorna la transcripción del audio.
    """
    start_time = time.time()
    logger.info(f"Received audio file: {file.filename}, size: {getattr(file, 'size', 'unknown')} bytes")
    
    temp_path = None
    
    try:
        # Crear directorio para archivos temporales si no existe
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generar nombre seguro para el archivo temporal
        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
        
        # Guardar archivo temporal con manejo de bloques para archivos grandes
        with open(temp_path, "wb") as temp_file:
            # Definir tamaño de bloque para lectura (1MB)
            chunk_size = 1024 * 1024
            file_content = await file.read(chunk_size)
            while file_content:
                temp_file.write(file_content)
                file_content = await file.read(chunk_size)
        
        logger.info(f"Audio file saved to temporary location: {temp_path}")
        
        # Obtener modelo con reintentos
        model = None
        max_retries = 3
        retry_delay = 1  # segundos
        last_error = None
        
        for attempt in range(max_retries):
            try:
                model = get_model()
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Error loading model (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if model is None:
            logger.error(f"Failed to load model after {max_retries} attempts")
            raise HTTPException(
                status_code=500,
                detail=f"Error al cargar el modelo después de {max_retries} intentos: {str(last_error)}"
            )
        
        # Configurar opciones de transcripción
        options = {}
        if language:
            options["language"] = language
        
        # Transcribir audio con timeout
        max_processing_time = 300  # 5 minutos
        
        try:
            with timeout(seconds=max_processing_time):
                logger.info(f"Starting transcription of file: {temp_path}")
                result = model.transcribe(temp_path, **options)
        except TimeoutError:
            logger.error(f"Timeout transcribing audio file: {temp_path}")
            raise HTTPException(
                status_code=408,
                detail=f"Tiempo de procesamiento excedido ({max_processing_time}s). El archivo puede ser demasiado largo."
            )
        
        transcription = result["text"]
        detected_language = result.get("language", "unknown")
        language_probability = result.get("language_probability", 0.0)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        
        return {
            "transcription": transcription,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "processing_time_seconds": elapsed_time
        }
    
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la transcripción: {str(e)}")
    finally:
        # Limpiar archivo temporal
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Deleted temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Error deleting temporary file {temp_path}: {str(e)}")

@app.post("/transcribe-async")
async def transcribe_audio_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_audio_file),
    language: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    """
    Procesa la transcripción de audio de forma asíncrona.
    
    - **file**: Archivo de audio a transcribir
    - **language**: Código de idioma opcional
    - **callback_url**: URL opcional para recibir el resultado via webhook
    
    Retorna un ID de tarea que puede usarse para verificar el estado posteriormente.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Starting async transcription task {task_id} for file {file.filename}")
    
    # Crear directorio para archivos temporales si no existe
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Guardar archivo temporal
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(temp_dir, f"{task_id}{file_ext}")
    
    try:
        # Guardar archivo con manejo de bloques para archivos grandes
        with open(temp_path, "wb") as temp_file:
            # Definir tamaño de bloque para lectura (1MB)
            chunk_size = 1024 * 1024
            file_content = await file.read(chunk_size)
            while file_content:
                temp_file.write(file_content)
                file_content = await file.read(chunk_size)
        
        # Registrar tarea
        transcription_tasks[task_id] = {
            "status": "processing",
            "started_at": time.time(),
            "file_path": temp_path,
            "callback_url": callback_url,
            "language": language,
            "original_filename": file.filename
        }
        
        # Iniciar procesamiento en segundo plano
        background_tasks.add_task(
            process_transcription,
            task_id,
            temp_path,
            language,
            callback_url
        )
        
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        # Limpiar en caso de error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        logger.error(f"Error setting up async transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar la transcripción asíncrona: {str(e)}")

async def process_transcription(
    task_id: str,
    file_path: str,
    language: Optional[str],
    callback_url: Optional[str]
):
    """Tarea en segundo plano para procesar la transcripción"""
    if task_id not in transcription_tasks:
        logger.error(f"Task {task_id} not found in transcription_tasks")
        return
        
    try:
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo temporal {file_path} no existe")
            
        # Obtener modelo con reintentos
        model = None
        max_retries = 3
        retry_delay = 1  # segundos
        
        for attempt in range(max_retries):
            try:
                model = get_model()
                break
            except Exception as e:
                logger.warning(f"Error loading model for task {task_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if model is None:
            raise RuntimeError(f"No se pudo cargar el modelo después de {max_retries} intentos")
        
        # Configurar opciones
        options = {}
        if language:
            options["language"] = language
        
        # Transcribir audio con timeout
        max_processing_time = 600  # 10 minutos para procesamiento asíncrono
        
        try:
            with timeout(seconds=max_processing_time):
                logger.info(f"Starting async transcription of file: {file_path}")
                result = model.transcribe(file_path, **options)
        except TimeoutError:
            raise TimeoutError(f"Tiempo de procesamiento excedido ({max_processing_time}s)")
        
        transcription = result["text"]
        detected_language = result.get("language", "unknown")
        
        # Actualizar estado
        transcription_tasks[task_id].update({
            "status": "completed",
            "transcription": transcription,
            "detected_language": detected_language,
            "completed_at": time.time(),
            "processing_time": time.time() - transcription_tasks[task_id]["started_at"]
        })
        
        logger.info(f"Task {task_id} completed successfully")
        
        # Callback opcional
        if callback_url:
            try:
                # Aquí implementaríamos lógica para enviar callback
                # (requeriría importar aiohttp u otra biblioteca HTTP)
                logger.info(f"Callback would be sent to {callback_url}")
            except Exception as callback_error:
                logger.error(f"Error sending callback for task {task_id}: {str(callback_error)}")
        
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
        transcription_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        })
    finally:
        # Limpiar archivo temporal
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted temporary file for task {task_id}: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Error deleting temporary file {file_path}: {str(cleanup_error)}")

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Verificar el estado de una tarea de transcripción asíncrona
    
    - **task_id**: ID de la tarea retornado por el endpoint /transcribe-async
    
    Retorna el estado actual y el resultado si está disponible.
    """
    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail=f"Tarea no encontrada: {task_id}")
    
    # Crear una copia para evitar mutaciones concurrentes
    task = dict(transcription_tasks[task_id])
    
    # No retornar la ruta del archivo
    if "file_path" in task:
        del task["file_path"]
        
    return task

@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud
    
    Retorna información sobre el estado del servicio y el modelo cargado.
    """
    model_loaded = _model is not None
    
    # Verificar dependencias
    dependencies_status = {}
    for dep in ["fastapi", "python-multipart", "whisper"]:
        dependencies_status[dep] = check_dependency(dep)
    
    model_name = "unknown"
    if model_loaded:
        try:
            model_name = _model.model_name
        except:
            model_name = os.getenv("MODEL_SIZE", "base")
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": model_name,
        "model_loaded": model_loaded,
        "dependencies": dependencies_status,
        "tasks_in_progress": sum(1 for task in transcription_tasks.values() if task.get("status") == "processing"),
        "total_tasks": len(transcription_tasks)
    }

@app.get("/supported-languages")
async def get_supported_languages():
    """
    Obtener la lista de idiomas soportados por Whisper
    
    Retorna códigos y nombres de idiomas soportados.
    """
    # Idiomas soportados por Whisper
    languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "nl", "name": "Dutch"},
        {"code": "ja", "name": "Japanese"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ru", "name": "Russian"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ko", "name": "Korean"},
        {"code": "tr", "name": "Turkish"},
        {"code": "pl", "name": "Polish"},
        {"code": "sv", "name": "Swedish"},
        {"code": "da", "name": "Danish"},
        {"code": "fi", "name": "Finnish"},
        {"code": "no", "name": "Norwegian"},
    ]
    
    return {"languages": languages}

# Manejo global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no controlado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno del servidor: {str(exc)}"}
    )

# Manejo de limpieza al iniciar/detener la aplicación
@app.on_event("startup")
async def startup_event():
    """Tareas a ejecutar al iniciar la aplicación"""
    logger.info("Iniciando el servicio de transcripción")
    
    # Limpiar archivos temporales antiguos
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    if os.path.exists(temp_dir):
        try:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            logger.info(f"Limpieza de archivos temporales completada")
        except Exception as e:
            logger.warning(f"Error al limpiar archivos temporales: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Tareas a ejecutar al detener la aplicación"""
    logger.info("Deteniendo el servicio de transcripción")
    
    # Limpiar archivos temporales
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    if os.path.exists(temp_dir):
        try:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            logger.info(f"Limpieza de archivos temporales completada")
        except Exception as e:
            logger.warning(f"Error al limpiar archivos temporales: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Verificar dependencias críticas
    try:
        import fastapi
        import multipart
    except ImportError as e:
        print(f"Error: Faltan dependencias críticas. {str(e)}")
        print("Instale las dependencias requeridas con:")
        print("pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    
    # Obtener puerto de variable de entorno o usar 8000
    port = int(os.getenv("PORT", "8000"))
    
    # Ejecutar servidor
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

