"""
API endpoints for continuous learning pipeline
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import tempfile
import logging
import json
from datetime import datetime
import pandas as pd

from .pipeline import ContinuousLearningPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Continuous Learning API",
    description="API for BERT product classification with continuous learning",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

# Request/Response models
class PredictionRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 32

class PredictionResponse(BaseModel):
    predictions: List[Dict]
    processing_time: float

class TrainingRequest(BaseModel):
    strategy: str = "knowledge_distillation"
    old_validation_data_path: Optional[str] = None

class TrainingResponse(BaseModel):
    success: bool
    training_id: str
    message: str
    results: Optional[Dict] = None

class ModelInfoResponse(BaseModel):
    model_info: Dict
    status: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the continuous learning pipeline on startup"""
    global pipeline
    try:
        pipeline = ContinuousLearningPipeline()
        pipeline.initialize()
        logging.info("Continuous Learning Pipeline initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline and pipeline.is_initialized:
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions on input texts"""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = datetime.now()
        batch_size = request.batch_size if request.batch_size is not None else 32
        predictions = pipeline.predict(request.texts, batch_size)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            predictions=predictions,
            processing_time=processing_time
        )
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model info endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model"""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        model_info = pipeline.get_model_info()
        return ModelInfoResponse(
            model_info=model_info,
            status="active"
        )
    except Exception as e:
        logging.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Training endpoint with file upload
@app.post("/train/incremental", response_model=TrainingResponse)
async def train_incremental(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str = "knowledge_distillation",
    old_validation_data_path: Optional[str] = None
):
    """Trigger incremental training with uploaded data"""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    if not file.filename or not file.filename.endswith(('.csv', '.parquet')):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
    
    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Generate training ID
        training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start training in background
        background_tasks.add_task(
            run_incremental_training,
            tmp_file_path,
            strategy,
            old_validation_data_path,
            training_id
        )
        
        return TrainingResponse(
            success=True,
            training_id=training_id,
            message="Incremental training started in background"
        )
        
    except Exception as e:
        logging.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

async def run_incremental_training(data_path: str, strategy: str, 
                                 old_validation_data_path: Optional[str], training_id: str):
    """Run incremental training in background"""
    global pipeline
    try:
        logging.info(f"Starting background training {training_id}")
        
        if not pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        results = pipeline.train_incremental(
            data_path,
            strategy=strategy,
            old_validation_data_path=old_validation_data_path
        )
        
        # Save training results
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        results_file = os.path.join(results_dir, f"{training_id}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Training {training_id} completed successfully")
        
        # Clean up temporary file
        if os.path.exists(data_path):
            os.unlink(data_path)
            
    except Exception as e:
        logging.error(f"Background training {training_id} failed: {e}")
    finally:
        # Clean up temporary file in case of error
        if os.path.exists(data_path):
            os.unlink(data_path)

# Training status endpoint
@app.get("/train/status/{training_id}")
async def get_training_status(training_id: str):
    """Get the status of a training job"""
    results_file = f"training_results/{training_id}_results.json"
    
    if os.path.exists(results_file):
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        return {"status": "completed", "results": results}
    else:
        return {"status": "running", "message": "Training in progress"}

# Export model endpoint
@app.post("/model/export")
async def export_model(output_path: str):
    """Export the current model for deployment"""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.export_model_for_deployment(output_path)
        return {"success": True, "message": f"Model exported to {output_path}"}
    except Exception as e:
        logging.error(f"Model export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")

# Batch prediction endpoint for large datasets
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Make predictions on a batch of data from uploaded file"""
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not file.filename or not file.filename.endswith(('.csv', '.parquet')):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")
    
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load data
        if file.filename.endswith('.csv'):
            data = pd.read_csv(tmp_file_path)
        else:
            data = pd.read_parquet(tmp_file_path)
        
        # Check for required columns
        text_column = None
        for col in ['product_name', 'input_text', 'text', 'description']:
            if col in data.columns:
                text_column = col
                break
        
        if text_column is None:
            raise HTTPException(status_code=400, detail="No recognized text column found")
        
        # Make predictions
        texts = data[text_column].astype(str).tolist()
        predictions = pipeline.predict(texts, batch_size=64)
        
        # Add predictions to dataframe
        data['predicted_category'] = [p['predicted_label'] for p in predictions]
        data['confidence'] = [p['confidence'] for p in predictions]
        
        # Save results
        results_dir = "batch_predictions"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(results_dir, f"predictions_{timestamp}.csv")
        data.to_csv(output_file, index=False)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "success": True,
            "message": f"Batch predictions completed",
            "output_file": output_file,
            "num_predictions": len(predictions)
        }
        
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Configuration endpoints
@app.get("/config")
async def get_config():
    """Get current configuration"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.config.to_dict()

@app.post("/config/update")
async def update_config(config_updates: Dict):
    """Update configuration parameters"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Update configuration
        for key, value in config_updates.items():
            if hasattr(pipeline.config, key):
                setattr(pipeline.config, key, value)
        
        # Save updated configuration
        pipeline.config.save_to_yaml("continuous_learning_config.yaml")
        
        return {"success": True, "message": "Configuration updated successfully"}
    except Exception as e:
        logging.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

# Cleanup endpoint
@app.post("/cleanup/checkpoints")
async def cleanup_checkpoints(keep_last_n: int = 5):
    """Clean up old checkpoints"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        pipeline.cleanup_old_checkpoints(keep_last_n)
        return {"success": True, "message": f"Cleanup completed. Kept last {keep_last_n} checkpoints"}
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
