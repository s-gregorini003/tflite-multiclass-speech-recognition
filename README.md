# Multiclass Speech Recognition with TensorFlow Lite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-gregorini003/tflite-multiclass-speech-recognition/blob/master/tflite_multiclass_speech_recognition.ipynb)

This project demonstrates how to use TensorFlow and Keras to train three different CNNs to perform multiclass keyword spotting. Then, the trained model can be used to run inference on a Raspberry Pi and control a smart light with voice commands.




## Usage

## Dataset

The dataset used for training and validation is the [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). It consists of 105 829 audio samples of 35 different words from 2 618 speakers. The length of the utterances is 1 s, they are recorded with a sample rate of 16 kHz and stored as 16-bit, single channel WAVE files. To train the networks with a less skewed dataset, 70% of the samples from the unselected classes are discarded. Therefore, the training set contains 44 926 audio files divided into 7 classes (6 selected keywords + 1 "unknown word" class).

The keywords are:
- on
- off
- up
- down
- bed
- visual

## Models

The CNNs selected are [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Krizhevsky et al., 2012), [VGG16](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan and Zisserman, 2015) and [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf) (Iandola et al., 2016). Since they were designed for computer vision, some changes in the models' architecture are made.


## Test the Model in Google Colab

## Run Inference on a Raspberry Pi

### Dependencies

To deploy the trained model you need the TensorFlow Lite inference engine installed on your Raspberry Pi. Instructions to do that can be found here: https://www.tensorflow.org/lite/guide/python. Additionally, the following Python modules are required:

- `sounddevice`
- `numpy`
- `scipy.signal`
- `timeit`
- `python_speech_features`
- `importlib`

These can be installed using `pip`.


### Yeelight LED Bulb 1S Configuration

## Credits

This project was created starting from Shawn Hymel's TensorFlow Lite Tutorials ([Part 1](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-1-wake-word-feature-extraction/54e1ce8520154081a58feb301ef9d87a), [Part 2](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-2-speech-recognition-model-training/d8d04a2b60a442cf8c3fa5c0dd2a292b), [Part 3](https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-3-speech-recognition-on-raspberry-pi/8a2dc7d8a9a947b4a953d37d3b271c71)). A big thanks goes to Shawn for his awesome and detailed work.


## License

All code in this repository is for demonstration purposes and licensed under [MIT License](https://en.wikipedia.org/wiki/MIT_License).

Distributed as-is. No warranty is given.
