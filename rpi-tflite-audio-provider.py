import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO

import tflite_runtime.interpreter as tflite

import importlib
nc = importlib.import_module("nc-message-sender")

# Parameters
debug_time = 0
debug_acc = 0
word_threshold = 0.95
rec_duration = 0.5	 # 0.5
sample_length = 0.5
window_stride = 0.5	 # 0.5
sample_rate = 8000	 # The mic requires at least 44100 Hz to work
resample_rate = 8000
num_channels = 1
num_mfcc = 16

model_path = '07_14-44_alexnet_wake_word_model_lite.tflite'
#model_path = '12_14-33_vgg16_wake_word_model_lite.tflite'
#model_path = '07_18-36_squeezenet_wake_word_model_lite.tflite'

hostname = "192.168.1.11"
port = 55443

# Sliding window
# window = np.zeros(int(sample_length * resample_rate) * 2)

mfccs_old = np.zeros((32, 25))

# Load model (interpreter)
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Filter and downsample
def decimate(signal, old_fs, new_fs):

    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # Downsampling is possible only by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only downsample by integer factor")

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

def msgSwitcher(i):
    switcher = {
        0:"{\"id\":1,\"method\":\"set_rgb\",\"params\":[255,\"smooth\",500]}\r\n",		    # bed
        1:"{\"id\":1,\"method\":\"set_bright\",\"params\":[100,\"smooth\",500]}\r\n",		# up
        2:"{\"id\":1,\"method\":\"set_power\",\"params\":[\"off\",\"smooth\",500]}\r\n",	# off
        3:"{\"id\":1,\"method\":\"set_rgb\",\"params\":[16711680,\"smooth\",500]}\r\n",		# visual
        4:"{\"id\":1,\"method\":\"set_bright\",\"params\":[10,\"smooth\",500]}\r\n",		# down
        5:"{\"id\":1,\"method\":\"set_power\",\"params\":[\"on\",\"smooth\",500,1]}\r\n"	# on
    }
    return switcher.get(i, "Unknown")

# Callback that gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for debug purposes
    start = timeit.default_timer()

    # Notify errors
    if status:
        print('Error:', status)

    # Remove second dimension from recording sample
    rec = np.squeeze(rec)

    # Decimate
#    rec, new_fs = decimate(rec, sample_rate, resample_rate)

    #  Store recording onto sliding window
#    stride = int(len(window)*window_stride)
#    window[:len(window)-stride] = window[stride:]
#    window[len(window)-stride:] = rec

    global mfccs_old

    # Compute MFCCs
    mfccs = python_speech_features.base.mfcc(rec,
                                            samplerate=resample_rate,
                                            winlen=0.02,
                                            winstep=0.02,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=512, # 2048
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=True,
                                            winfunc=np.hanning)

    delta = python_speech_features.base.delta(mfccs, 2)

    mfccs_delta = np.append(mfccs, delta, axis=1)

    mfccs_new = mfccs_delta.transpose()
    mfccs = np.append(mfccs_old, mfccs_new, axis=1)
#    mfccs = np.insert(mfccs, [0], 0, axis=1)
    mfccs_old = mfccs_new

    # Run inference and make predictions
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = np.amax(output_data)						# DEFINED FOR BINARY CLASSIFICATION, CHANGE TO MULTICLASS
    ind = np.where(output_data == val)
    prediction = ind[1].astype(int)
    if val > word_threshold:
        print('index:', ind[1])
        print('accuracy', val, '/n')
        msg = msgSwitcher(int(prediction))
#        print(msg)
        nc.netcat(hostname, port, msg.encode())
    if debug_acc:
#        print('accuracy:', val)
#        print('index:', ind[1])
        print('out tensor:', output_data)
    if debug_time:
        print(timeit.default_timer() - start)

# Start recording from microphone
with sd.InputStream(channels=num_channels,
        samplerate=sample_rate,
        blocksize=int(sample_rate * rec_duration),
        callback=sd_callback):
    while True:
        pass
