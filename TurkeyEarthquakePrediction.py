#!/usr/bin/env python
# coding: utf-8

# # Turkey Earthquake Prediction with Deep Learning Algorithm

# ## Import Libraries & Framework

# In[1]:


import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from tqdm import tqdm
import os

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Bidirectional

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf

from src.utils import get_project_config

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action="ignore")


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Project Config & Set Variables

# In[3]:


# Load Project Config
project_cfg = get_project_config("config.json",)
project_cfg


# In[4]:


# Set Constant Variables

RAW_DATA_PATH = project_cfg["data"]["rawData"]
# RAW_DATA_PATH

PREPROCESSED_DATA_PATH = project_cfg["data"]["processedData"]
# PREPROCESSED_DATA_PATH

RESULT_DATA_PATH = project_cfg["data"]["resultData"]
# RESULT_DATA_PATH

REPORTS_PATH = project_cfg["reports"]["folderName"]
# REPORTS_PATH

MODELS_PATH = project_cfg["models"]["folderName"]
# MODELS_PATH


# In[5]:


# Create Some Directories
os.makedirs(name=MODELS_PATH, exist_ok=True)
os.makedirs(name=REPORTS_PATH, exist_ok=True)


# ## Data Load

# In[6]:


# Read Data
df_data = pd.read_csv(os.path.join(RAW_DATA_PATH, "TurkeyEarthquakeData.xls"))
df_data


# In[7]:


# Show Data Shape Info
df_data.shape


# In[8]:


# Show Data Information
df_data.info()


# In[9]:


# Copy df_data
df_data_original = df_data.copy()


# In[10]:


# Choose Using features
df_data = df_data[['time', 'Latitude', 'Longitude', 'Depth', 'Magnitude' ]]


# In[11]:


# Rename Column Name & Convert Datetime Format
df_data.rename(columns={'time':'Time'}, inplace=True)
df_data

df_data['Time'] = pd.to_datetime(df_data['Time'])
df_data['Time'] = df_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_data


# In[12]:


# Set Index with Time Column
df_data.set_index('Time', inplace=True)
df_data


# In[13]:


# Sort Data with Index
df_data.sort_index(inplace=True)
df_data


# In[14]:


# Remove Duplicate Rows
df_data = df_data[~df_data.index.duplicated(keep='first')]
df_data


# In[15]:


# Index convert type to DateTime Object
df_data.index = pd.DatetimeIndex(df_data.index)
df_data


# In[16]:


# Show Data Information
df_data.info()


# In[17]:


# Show Statistical Information of data
df_data.describe().T


# ### Data Profiling

# In[18]:


# Create Small Data Profile
df_data_profile = ProfileReport(df_data, title="Profiling Report", minimal=True)
df_data_profile.to_file(os.path.join(REPORTS_PATH, "TurkeyEarthquake-MinimalDataProfiling.html"))

# Create Data Profile
df_data_profile = ProfileReport(df_data, title="Profiling Report")
df_data_profile.to_file(os.path.join(REPORTS_PATH, "TurkeyEarthquake-DataProfiling.html"))


# ### Data Visualizations

# In[19]:


# Set Chart Params
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14
plt.style.use('ggplot')


# In[20]:


# Visualize Each Columns
df_data.plot(subplots=True, figsize=(20, 12), sharex=False, sharey=False)


# In[21]:


# Create Logaritmic Feature for Each Columns 
extended_data = df_data.copy()

for i in extended_data.columns:
    extended_data["Log." + i] = np.log(extended_data[i] + 0.01)


# In[22]:


extended_data


# In[23]:


# Visualize Scatter Plot for Each Columns
lines = int(((len(extended_data.columns) - 1) * (len(extended_data.columns))) / 8) + 1
k = 1
subplt = plt.figure(figsize=(16, 60))

for i in range(0, len(extended_data.columns) - 1):
    for j in range(i + 1, len(extended_data.columns) - 1):
        subplt.add_subplot(lines, 4, k)
        plt.scatter(extended_data[extended_data.columns[i]], extended_data[extended_data.columns[j]])
        plt.title("{} X {}".format(extended_data.columns[i], extended_data.columns[j]))
        plt.xlabel(extended_data.columns[i])
        plt.ylabel(extended_data.columns[j])
        k += 1

plt.tight_layout()
plt.show()


# In[24]:


# Visualize Correlation Matrix

extended_corr = extended_data.corr()

plt.figure(figsize=(10, 5))
sns.heatmap(extended_corr, vmin=-1, vmax=1, cmap="bwr", annot=True, linewidth=0.1)
plt.title("Parametre Correlation Matrix")
plt.show()


# In[25]:


# Visualize ACF Chart

fig, ax = plt.subplots(2, 2, figsize=(20, 6))
plot_acf(df_data['Latitude'], ax=ax[0, 0], lags=100, title='Latitude')
ax[0, 0].set_title('Latitude')
plot_acf(df_data['Longitude'], ax=ax[0, 1], lags=100, title='Longitude')
ax[0, 1].set_title('Longitude')
plot_acf(df_data['Magnitude'], ax=ax[1, 0], lags=100, title='Magnitude')
ax[1, 0].set_title('Magnitude')
plot_acf(df_data['Depth'], ax=ax[1, 1], lags=100, title='Depth')
ax[1, 1].set_title('Depth')


# In[26]:


df_data


# ### Data Splitting

# In[27]:


# Split Data for Train & Test

df_train_data = df_data.iloc[:6000]
df_test_data = df_data.iloc[6000 - 12:]
df_test_original = df_data.iloc[6000:]

N_INPUT = 12
N_FEATURES = 4
BATCH_SIZE = 1
EPOCH_SIZE = 500


# In[28]:


df_train_data


# In[29]:


df_test_data


# In[30]:


# Show Shape of Dataframes 
df_train_data.shape, df_test_data.shape, df_test_original.shape


# In[31]:


# Create TimeSeriesGenerator for Training Data
train_generator = TimeseriesGenerator(df_train_data.values, df_train_data.values, length=N_INPUT, batch_size=BATCH_SIZE)


# In[32]:


# Create TimeSeriesGenerator for Test Data
test_generator = TimeseriesGenerator(df_test_data.values, df_test_data.values, length=N_INPUT, batch_size=BATCH_SIZE)


# In[33]:


# Split Train and Test Data
train_x = np.array([])
train_y = np.array([])
test_x = np.array([])
test_y = np.array([])

for i in range(len(train_generator)):
    a, b = train_generator[i]
    train_x = np.append(train_x, a.flatten())
    train_y = np.append(train_y, b)

for i in range(len(test_generator)):
    a, b = test_generator[i]
    test_x = np.append(test_x, a.flatten())
    test_y = np.append(test_y, b)


# In[34]:


# Reshape Data Dimension Shape
train_x = train_x.reshape(-1, N_FEATURES, N_INPUT)
train_y = train_y.reshape(-1, N_FEATURES)
test_x = test_x.reshape(-1, N_FEATURES, N_INPUT)
test_y = test_y.reshape(-1, N_FEATURES)


# In[35]:


# Show the shape of the Train & Test Data
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)


# ## Model Selection & Training

# ### LSTM Model

# In[36]:


# Create LSTM Model Architecture
model_lstm = Sequential()
model_lstm.add(LSTM(64, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=True))
model_lstm.add(LSTM(32, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=False))
model_lstm.add(Dense(32, activation='elu'))
model_lstm.add(Dense(16, activation='gelu'))
model_lstm.add(Dense(train_y.shape[1]))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.summary()


# In[37]:


# Create Directory for LSTM Model 
MODEL_DIR = project_cfg["models"]["folderName"]
MODEL_PATH = os.path.join(MODEL_DIR, "LSTM")

os.makedirs(name=MODEL_PATH, exist_ok=True)


# In[38]:


# Delete all previously saved model files
for i in os.listdir(MODEL_PATH):
    file_path = os.path.join(MODEL_PATH, i)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# In[39]:


# Save Model Weight per Epoch

model_dir_per_epoch = os.path.join(MODEL_PATH, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(model_dir_per_epoch, 
                             monitor='val_loss', 
                             save_best_only=False,
                             verbose=0)

callbacks = [checkpoint]


# In[40]:


# Train LSTM Model 

history = {'loss':[], 'val_loss':[]}

with tqdm(total=EPOCH_SIZE) as pbar:
    for i in range(EPOCH_SIZE):
        h = model_lstm.fit(train_x, train_y,
                           epochs=1,
                           batch_size=16,
                           validation_split=0.1,
                           callbacks=callbacks,
                           shuffle=False,
                           verbose=0)
        history['loss'].append(h.history['loss'])
        history['val_loss'].append(h.history['val_loss'])
        pbar.update(1)


# In[41]:


#  Visualization of Loss Values

print('Last Epoch Values: ')
print('loss: ', history['loss'][-1])
print('val_loss: ', history['val_loss'][-1])

loss_per_epoch = history['loss']
val_loss_per_epoch = history['val_loss']

# Visualizing Loss Values and MAE in each epoch
plt.plot(loss_per_epoch, label='Training Loss')
plt.plot(val_loss_per_epoch, label='Validation Loss')
plt.legend()
plt.show()


# In[42]:


# Load Best Performance Model

models_dir = os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[-5])
model = keras.models.load_model(models_dir)


# In[43]:


# Prediction Test Set

predictions = model.predict(test_x)
predictions_lstm = pd.DataFrame(predictions, columns=df_data[['Latitude', 'Longitude', 'Depth', 'Magnitude']].columns, index=df_test_original.index)
predictions_lstm


# In[44]:


# Calculate MSE, RMSE and MAE performance metrics

mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
rmse = np.sqrt(mse)
print('MSE: ', mse)
print('RMSE: ', rmse)
print('MAE: ', mae)


# In[46]:


# Compare Actual and Prediction Values

fig, ax = plt.subplots(2, 2, figsize=(10, 4))
for i, col in enumerate(predictions_lstm.columns):
    ax[i//2, i%2].plot(df_test_original[col], label='True')
    ax[i//2, i%2].plot(predictions_lstm[col], label='Predicted')
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].legend()
fig.suptitle('LSTM Model', fontsize=16)
plt.tight_layout()


# In[ ]:





# ### RNN Model

# In[47]:


# Create RNN Model Architecture

model_rnn = Sequential()
model_rnn.add(SimpleRNN(64, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=True))
model_rnn.add(SimpleRNN(32, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=False))
model_rnn.add(Dense(32, activation='elu'))
model_rnn.add(Dense(16, activation='gelu'))
model_rnn.add(Dense(train_y.shape[1]))
model_rnn.compile(optimizer='adam', loss='mse')

model_rnn.summary()


# In[48]:


# Create Directory for RNN Model

MODEL_PATH = os.path.join(MODEL_DIR, "RNN")
os.makedirs(name=MODEL_PATH, exist_ok=True)


# In[49]:


# Delete all previously saved model files
for i in os.listdir(MODEL_PATH):
    file_path = os.path.join(MODEL_PATH, i)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# In[50]:


# Save Model Weight per Epoch
model_dir_per_epoch = os.path.join(MODEL_PATH, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(model_dir_per_epoch, 
                             monitor='val_loss', 
                             save_best_only=False,
                             verbose=0)

callbacks = [checkpoint]


# In[51]:


# Train RNN Model

history = {'loss':[], 'val_loss':[]}

with tqdm(total=EPOCH_SIZE) as pbar:
    for i in range(EPOCH_SIZE):
        h = model_rnn.fit(train_x, train_y,
                          epochs=1,
                          batch_size=16,
                          validation_split=0.1,
                          callbacks=callbacks,
                          shuffle=False,
                          verbose=0)
        history['loss'].append(h.history['loss'])
        history['val_loss'].append(h.history['val_loss'])
        pbar.update(1)


# In[52]:


# Visualization of Loss Values

print('Last Epoch Values: ')
print('loss: ', history['loss'][-1])
print('val_loss: ', history['val_loss'][-1])

loss_per_epoch = history['loss']
val_loss_per_epoch = history['val_loss']


# In[53]:


# Visualizing Loss Values and MAE in each epoch

plt.plot(loss_per_epoch, label='Training Loss')
plt.plot(val_loss_per_epoch, label='Validation Loss')
plt.legend()
plt.show()


# In[54]:


# Load Best Performance Model

models_dir = os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[-5])
model = keras.models.load_model(models_dir)


# In[55]:


# Prediction Test Set

predictions = model.predict(test_x)
predictions_rnn = pd.DataFrame(predictions, columns=df_data.columns, index=df_test_original.index)
predictions_rnn


# In[56]:


# Calculate MSE, RMSE and MAE performance metrics

mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
rmse = np.sqrt(mse)
print('MSE: ', mse)
print('RMSE: ', rmse)
print('MAE: ', mae)


# In[57]:


# Compare Actual and Prediction Values

fig, ax = plt.subplots(2, 2, figsize=(10, 4))
for i, col in enumerate(predictions_rnn.columns):
    ax[i//2, i%2].plot(df_test_original[col], label='True')
    ax[i//2, i%2].plot(predictions_rnn[col], label='Predicted')
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].legend()
fig.suptitle('RNN Model', fontsize=16)
plt.tight_layout()


# In[ ]:





# ### GRU Model

# In[58]:


# Create GRU Model Architecture

model_gru = Sequential()
model_gru.add(GRU(64, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=True))
model_gru.add(GRU(32, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=False))
model_gru.add(Dense(32, activation='elu'))
model_gru.add(Dense(16, activation='gelu'))
model_gru.add(Dense(train_y.shape[1]))

model_gru.compile(optimizer='adam', loss='mse')

model_gru.summary()


# In[59]:


# Create Directory for GRU Model

MODEL_PATH = os.path.join(MODEL_DIR, "GRU")
os.makedirs(name=MODEL_PATH, exist_ok=True)


# In[60]:


# Delete all previously saved model files
for i in os.listdir(MODEL_PATH):
    file_path = os.path.join(MODEL_PATH, i)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# In[61]:


# Save Model Weight per Epoch
model_dir_per_epoch = os.path.join(MODEL_PATH, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(model_dir_per_epoch, 
                             monitor='val_loss', 
                             save_best_only=False,
                             verbose=0)

callbacks = [checkpoint]


# In[62]:


# Train GRU Model

history = {'loss':[], 'val_loss':[]}

with tqdm(total=EPOCH_SIZE) as pbar:
    for i in range(EPOCH_SIZE):
        h = model_gru.fit(train_x, train_y,
                          epochs=1,
                          batch_size=16,
                          validation_split=0.1,
                          callbacks=callbacks,
                          shuffle=False,
                          verbose=0)
        history['loss'].append(h.history['loss'])
        history['val_loss'].append(h.history['val_loss'])
        pbar.update(1)


# In[64]:


# Visualization of Loss Values

print('Last Epoch Values: ')
print('loss: ', history['loss'][-1])
print('val_loss: ', history['val_loss'][-1])

loss_per_epoch = history['loss']
val_loss_per_epoch = history['val_loss']


# In[65]:


# Visualizing Loss Values and MAE in each epoch

plt.plot(loss_per_epoch, label='Training Loss')
plt.plot(val_loss_per_epoch, label='Validation Loss')
plt.legend()
plt.show()


# In[66]:


# Load Best Performance Model

# models_dir = os.path.join(model_dir, os.listdir(model_dir)[-200])
models_dir = os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[-200])
model = keras.models.load_model(models_dir)


# In[67]:


# Prediction Test Set

predictions = model.predict(test_x)
predictions_gru = pd.DataFrame(predictions, columns=df_data.columns, index=df_test_original.index)
predictions_gru


# In[68]:


# Calculate MSE, RMSE and MAE performance metrics

mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
rmse = np.sqrt(mse)
print('MSE: ', mse)
print('RMSE: ', rmse)
print('MAE: ', mae)


# In[69]:


# Compare Actual and Prediction Values

fig, ax = plt.subplots(2, 2, figsize=(10, 4))
for i, col in enumerate(predictions_gru.columns):
    ax[i//2, i%2].plot(df_test_original[col], label='True')
    ax[i//2, i%2].plot(predictions_gru[col], label='Predicted')
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].legend()
fig.suptitle('GRU Model', fontsize=16)
plt.tight_layout()


# In[ ]:





# ### Bidirectional Model

# In[70]:


# Create Bidirectional Model Architecture

model_bid_lstm = Sequential()
model_bid_lstm.add(Bidirectional(LSTM(64, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=True)))
model_bid_lstm.add(Bidirectional(LSTM(32, activation='tanh', input_shape=(N_FEATURES, N_INPUT), return_sequences=False)))
model_bid_lstm.add(Dense(32, activation='elu'))
model_bid_lstm.add(Dense(16, activation='gelu'))
model_bid_lstm.add(Dense(train_y.shape[1]))
model_bid_lstm.compile(optimizer='adam', loss='mse')

model_bid_lstm.build(input_shape=(None, N_FEATURES, N_INPUT))
model_bid_lstm.summary()


# In[71]:


# Create Directory for Bidirectional Model

MODEL_PATH = os.path.join(MODEL_DIR, "Bidirectional")
os.makedirs(name=MODEL_PATH, exist_ok=True)


# In[72]:


# Delete all previously saved model files
for i in os.listdir(MODEL_PATH):
    file_path = os.path.join(MODEL_PATH, i)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# In[73]:


# Save Model Weight per Epoch
model_dir_per_epoch = os.path.join(MODEL_PATH, 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(model_dir_per_epoch, 
                             monitor='val_loss', 
                             save_best_only=False,
                             verbose=0)

callbacks = [checkpoint]


# In[74]:


# Train Bidirectional Model
history = {'loss':[], 'val_loss':[]}

with tqdm(total=EPOCH_SIZE) as pbar:
    for i in range(EPOCH_SIZE):
        h = model_bid_lstm.fit(train_x, train_y,
                               epochs=1,
                               batch_size=16,
                               validation_split=0.1,
                               callbacks=callbacks,
                               shuffle=False,
                               verbose=0)
        history['loss'].append(h.history['loss'])
        history['val_loss'].append(h.history['val_loss'])
        pbar.update(1)


# In[75]:


# Visualization of Loss Values

print('Last Epoch Values: ')
print('loss: ', history['loss'][-1])
print('val_loss: ', history['val_loss'][-1])

loss_per_epoch = history['loss']
val_loss_per_epoch = history['val_loss']


# In[76]:


# Visualizing Loss Values and MAE in each epoch

plt.plot(loss_per_epoch, label='Training Loss')
plt.plot(val_loss_per_epoch, label='Validation Loss')
plt.legend()
plt.show()


# In[77]:


# Load Best Performance Model

# models_dir = os.path.join(model_dir, os.listdir(model_dir)[-10])
models_dir = os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[-10])
model = keras.models.load_model(models_dir)


# In[78]:


# Prediction Test Set

predictions = model.predict(test_x)
predictions_bid_lstm = pd.DataFrame(predictions, columns=df_data.columns, index=df_test_original.index)
predictions_bid_lstm


# In[79]:


# Calculate MSE, RMSE and MAE performance metrics

mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
rmse = np.sqrt(mse)
print('MSE: ', mse)
print('RMSE: ', rmse)
print('MAE: ', mae)


# In[80]:


# Compare Actual and Prediction Values

fig, ax = plt.subplots(2, 2, figsize=(10, 4))
for i, col in enumerate(predictions_bid_lstm.columns):
    ax[i//2, i%2].plot(df_test_original[col], label='True')
    ax[i//2, i%2].plot(predictions_bid_lstm[col], label='Predicted')
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].legend()
fig.suptitle('Bidirectional Model', fontsize=16)
plt.tight_layout()


# ### Evaluating Models Output

# In[81]:


fig, ax = plt.subplots(2, 2, figsize=(20, 10))
for i, col in enumerate(predictions_gru.columns):
    ax[i//2, i%2].plot(df_test_original[col], label='True')
    ax[i//2, i%2].plot(predictions_lstm[col], label='LSTM Predicted')
    ax[i//2, i%2].plot(predictions_rnn[col], label='RNN Predicted')
    ax[i//2, i%2].plot(predictions_gru[col], label='GRU Predicted')
    ax[i//2, i%2].plot(predictions_bid_lstm[col], label='BiLSTM Predicted')
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].legend()
plt.tight_layout()


# In[ ]:





# In[ ]:




