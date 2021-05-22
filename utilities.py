# Utilities Code
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data (json_file):
    with open(json_file, "r") as fp:
        data = json.load(fp)
        
        # Note: mfcc was converted from a numpy array to list before storing in JSON
        #  Need convert back to numpy array
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])
        
    return X, y

def load_mappings(json_file):
    with open(json_file, "r") as fp:
        data = json.load(fp)
        
        # Note: mfcc was converted from a numpy array to list before storing in JSON
        #  Need convert back to numpy array
        machine_audio_mapping = np.array(data["mapping"])
        
    return machine_audio_mapping 

def plot_history (history, regularization_flag=False):
    """Plot training history.

    Keyword arguments:
    history -- dictionary with loss and accuracy
    regularization_flag -- whether the plot is before or after regularization (default False)
    
    Returns:
    None
    """
    
    sns.set_theme(style="darkgrid")
    # Plot the responses for different events and regions

    fig, axs = plt.subplots(2, figsize=(12, 8)) # width, height
            
    # Accuracy sub-plot    
    axs[0].plot(history.history["acc"], label="Training accuracy")
    axs[0].plot(history.history["val_acc"], label="Test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_facecolor('#FFFFF0') # little darker biege: #F5F5DC
    if regularization_flag: axs[0].set_title("Accuracy plot (with regularizatiton)")
    else: axs[0].set_title("Accuracy plot")
    
    # Error (loss) sub-plot
    axs[1].plot(history.history["loss"], label="Training error")
    axs[1].plot(history.history["val_loss"], label="Test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epochs")   
    axs[1].legend(loc="upper right")
    axs[1].set_facecolor('#FFFFF0')
    if regularization_flag: axs[1].set_title("Error plot (with regularizatiton)")
    else: axs[1].set_title("Error plot")
    
    plt.show()
    return
    
def predict(model, X, y, json_file):
    
    # For prediction we need 4 D so need to add the sample number as 1st dim.
    # X -> (1, 130, 13, 1) where the 1st dim is the sample size = 1
    X = X[np.newaxis, ...]
    
    # Get "predictions", which is an array of probablities [[0.37, 0.11, 0.87, ...]]
    prediction = model.predict(X)
    
    # Extract index with max. value
    predicted_index = np.argmax(prediction, axis=1)

    # Get semantaic labels (i.e. text genre labels)
    mapping = load_mappings(json_file)
    
    # Map indexes to genre labels
    expected_machine_label = mapping[y]
    predicted_machine_label = mapping[predicted_index][0]
    
    print("Expected machine signal: '{}'. Predicted machine signal: '{}'".format(expected_machine_label, predicted_machine_label))
    
    return