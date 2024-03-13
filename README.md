# Turkey Earthquake Prediction with Deep Learning Algorithms
## Project Folder
- **`cfg/`**
    - project.json
- **`data/`**
    - **`raw/`**
- **`models/`**
    - **`LSTM/`**
        - weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5
    - **`RNN/`**
        - weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5
    - **`GRU/`**
        - weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5
    - **`Bidirectional/`**
        - weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5
- **`reports/`**
    - TurkeyEarthquake-MinimalDataProfiling.html
    - TurkeyEarthquake-DataProfiling.html

- **`src/`**
    - utils.py
- .gitignore
- TurkeyEarthquakePrediction.ipynb
- ReadMe.md
- ReadMe.pdf
- requirements.txt  

## Abstract
> Earthquakes, a natural phenomenon prevalent in seismically active regions, pose a grave threat to human lives and infrastructure. Despite today's technological advancements, a definitive earthquake prediction method remains elusive. However, researchers across diverse scientific fields are diligently studying past earthquake records in hopes of uncovering discernible patterns. To anticipate impending earthquakes, comprehensive investigations are conducted, drawing expertise from various disciplines. Notably, the rise of information technologies has steered attention towards deep learning, a subset of artificial intelligence, as a means to achieve accurate predictions in this complex process. In this research study, research is carried out on a possible future earthquake prediction model using deep learning architectures with the data of earthquakes that have occurred in Turkey. The topic of study includes a model investigated by using information such as time, latitude, longitude, magnitude and depth of the catalog data of earthquakes that have occurred in Turkey and deep learning algorithms. Long-Short Term Memory (LSTM) architecture, which is one of the Recurrent Neural Network processes, is used to predict the time of occurrence of the earthquake that may occur in the research. In the process of developing the model, the RNN model demonstrates superior prediction accuracy compared to the LSTM model. The RNN model achieves lower values for all evaluation metrics: MSE (95.13 vs. 195.54), RMSE (9.75 vs. 13.98), and MAE (4.70 vs. 5.71). This underscores the RNN model’s better performance, reflected by consistently reduced error metrics across all measures

## Installation
### 1.1.1 - Create New Env. with Conda Package Managemnet
```
conda create --name "forecast_env" python
conda activate forecast_env
```    

### 1.2- Install in an Actual Env.
```
pip install -r requirements.txt
```

## Run
> Open the `TurkeyEarthquakePrediction.ipynb` notebook and run it

## Project presentation upload video Link

> Use the link to access the project presentation. However, only those who have permission to access the National Collage of Ireland infrastructure can view the video. If you encounter a problem during the project stages, please get in touch.

## Contact
- Omer Kas - x20244100@student.ncirl.ie