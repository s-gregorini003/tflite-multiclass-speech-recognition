# Multiclass Speech Recognition with TensorFlow Lite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-gregorini003/tflite-multiclass-speech-recognition/blob/master/tflite_multiclass_speech_recognition.ipynb)

This project demonstrates how to use TensorFlow and Keras to train three different CNNs to perform multiclass keyword spotting. Then, the trained model can be used to run inference on a Raspberry Pi and control a smart light with voice commands. [This video](https://www.youtube.com/watch?v=BmgrIliMWqU) shows how the deployed system works on a Raspberry Pi 4.



### Dataset

The dataset used for training and validation is the [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). It consists of 105 829 audio samples of 35 different words from 2 618 speakers. The length of the utterances is 1 s, they are recorded with a sample rate of 16 kHz and stored as 16-bit, single channel WAVE files. To train the networks with a less skewed dataset, 70% of the samples from the unselected classes are discarded. Therefore, the training set contains 44 926 audio files divided into 7 classes (6 selected keywords + 1 "unknown word" class).

The following graph shows the number of samples included in each category, as well as the keywords used for the project.

![Keywords used for the project](https://github.com/s-gregorini003/tflite-multiclass-speech-recognition/blob/master/img/keywords.png)

### Models

The CNNs selected are [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Krizhevsky et al., 2012), [VGG16](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan and Zisserman, 2015) and [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf) (Iandola et al., 2016). Since they were designed for computer vision, some changes in the models' architecture are made.

![Investigated models architecture](https://github.com/s-gregorini003/tflite-multiclass-speech-recognition/blob/master/img/investigated-models.png)

## Try the Models in Colab

You can try the system in Colab without deploying the system on a Pi (obviously there won't be the smart light integration). First, download one of the pretrained models from these links:

- [alexnet_tflite_kws_model.h5](https://mega.nz/file/As8FmKaZ#tD19NuM20v6fICTVc9mlnCu96PbMyLs-y9RRCkfl744)

- [squeezenet_tflite_kws_model.h5](https://mega.nz/file/81VGla6C#rebzDLHpsvPoANFJB64g7t0J1PKxRftTLd88aU1fo2g)

- [vgg16_tflite_kws_model.h5](https://mega.nz/file/xskTHI6D#xjfoEvst9HWaFQsrBmXjXM_7dQzcf1MCX6TCNhYkoGE)

Then, open the notebook in Colab and upload the ".h5" file into the `/tmp/` folder. In the first cell of section **4 Interactive Testing**, follow the instruction and type the name of the model in the variable `selected_model`. Finally, run the very first cell of the notebook and then all the cells in section 4. 


## Run On-device Inference

To deploy the trained model you need the TensorFlow Lite inference engine installed on your Raspberry Pi. Instructions to do that can be found here: [Python quickstart](https://www.tensorflow.org/lite/guide/python). Additionally, the following Python modules are required:

- `sounddevice`
- `numpy`
- `scipy.signal`
- `timeit`
- `python_speech_features`
- `importlib`

All these packages can be installed through the `pip` command.


### Yeelight LED Bulb 1S Configuration

The first step to set up the lightbulb is to download the official app from [Yeelight](https://www.yeelight.com) and perform the initial configuration.

<a href="https://play.google.com/store/apps/details?id=com.yeelight.cherry&hl=it"><img alt="Get it on Google Play" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height=60px /></a>
<a href="https://apps.apple.com/it/app/yeelight/id977125608"><img alt="Download on the App Store" src="https://upload.wikimedia.org/wikipedia/commons/3/3c/Download_on_the_App_Store_Badge.svg" height=60px /></a>


After providing your router's SSID and password to the app, the light will be connected and visible to any device under the same network. After the configuration procedure is completed, it is possible to enable the third party control protocol directly in the app. This protocol allows the light to be managed locally (LAN connection).

![LAN control enable process](https://github.com/s-gregorini003/tflite-multiclass-speech-recognition/blob/master/img/lan-control-enabling.png)


Then, copy the IP address of the lightbulb (which can be found in the light settings of the app) into the `rpi-tflite-audio-provider.py`, specifically replace the string variable `hostname`. Copy the files `rpi-tflite-audio-provider.py` and `nc-message-sender.py` into a directory on the Pi and connect a USB microphone. Finally, open the terminal on the Pi, `cd` to the folder where you copied the files and run the script.

`python3 rpi-tflite-audio-provider.py`


## Credits

This project was created starting from Shawn Hymel's TensorFlow Lite Tutorials ([Part 1](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-1-wake-word-feature-extraction/54e1ce8520154081a58feb301ef9d87a), [Part 2](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-2-speech-recognition-model-training/d8d04a2b60a442cf8c3fa5c0dd2a292b), [Part 3](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-3-speech-recognition-on-raspberry-pi/8a2dc7d8a9a947b4a953d37d3b271c71)). A big thanks goes to Shawn for his awesome and detailed work.


## License

All code in this repository is for demonstration purposes and licensed under [MIT License](https://en.wikipedia.org/wiki/MIT_License).

Distributed as-is. No warranty is given.
