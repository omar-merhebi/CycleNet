import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

# // TODO: Finish inference logic

def inference(config, inference_data, model, cell_id_key='cell_id'):
    model_path = Path(config.mode.model)
    
    model = tf.keras.models.load_model(model_path)
    
    colnames = ['id', 'prediction', 'confidence']
    labels = inference_data.labels
        
    results_df = pd.DataFrame(columns=colnames)
        
    for s, step in enumerate(inference_data):
        ids = step[-1][cell_id_key].numpy()
        
        raw_predictions = model.predict(step[0])
        confidences = np.max(raw_predictions, axis=-1)
        predictions = np.argmax(raw_predictions, axis=-1)
        predictions = np.array([labels[i] for i in predictions])
        
        to_stack = [ids, predictions, confidences]
            
        stacked = np.vstack(to_stack)
        temp_df = pd.DataFrame(stacked.T, columns=colnames)
            
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
        return results_df

