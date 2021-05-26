# Utilities Code
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def load_data (json_file):
    with open(json_file, "r") as fp:
        data = json.load(fp)
        
        # Note: mfcc was converted from a numpy array to list before storing in JSON
        #  Need convert back to numpy array
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])
        
    return X, y

def map_signal_labels(json_file):
    """Return the semantic/actual label from 'index'. These are the folder names such as 'valve_abnormal' 

    Arguments:
    json_file -- Data dictionary that has the mapping to folder names 
    
    Returns:
    machine_label_mapping -- The mapped labels
    """
    with open(json_file, "r") as fp:
        data = json.load(fp)
        
        # Note: mfcc was converted from a numpy array to list before storing in JSON
        #  Need convert back to numpy array
        machine_label_mapping = np.array(data["mapping"])
        
    return machine_label_mapping 

def plot_history (history, regularization_flag=False):
    """Plot training history.

    Arguments:
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
    
def predict(model, X, y, signal_label_mappings):
    
    # For prediction we need 4 D so need to add the sample number as 1st dim.
    # X -> (1, 130, 13, 1) where the 1st dim is the sample size = 1
    X = X[np.newaxis, ...]
    
    # Get "predictions", which is an array of probablities [[0.37, 0.11, 0.87, ...]]
    prediction = model.predict(X)
    
    # Extract index with max. value
    predicted_index = np.argmax(prediction, axis=1)
    
    # Map indexes to genre labels
    actual_machine_label = signal_label_mappings[y]
    predicted_machine_label = signal_label_mappings[predicted_index][0]
    
    #print("Expected machine signal: '{}' \t\t|  Predicted: '{}'".
    #      format(actual_machine_label, predicted_machine_label))
    
    return (actual_machine_label, predicted_machine_label)

def plot_confusion_matrix(y_actual, y_predicted):
    data = confusion_matrix(y_actual, y_predicted)
    df_cm = pd.DataFrame(data, columns=np.unique(y_actual), index = np.unique(y_actual))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 10}) # font size
    return