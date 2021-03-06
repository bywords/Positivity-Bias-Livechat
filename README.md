# Positivity-Bias-Livechat
A keras implementation of deep learning approaches used in "Positivity Bias in Customer Satisfaction Ratings", published at the International Web Conference '2018 (TheWebConf, also known as WWW) Big Web Track [[pdf]](https://dl.acm.org/citation.cfm?id=3186579)

## Usage

```
python main.py
```

Before running the above script, it is required to specify threshold values used for time gap features at utils.py.


## Input data format

We are not able to release our dataset due to the permission issues. Alternatively, the input data format is shown below to assist potential users with running the code in other dataset.

### data/chat_sequences.txt
```
SessionID|Nth|Speaker|Utterance|Type|DateTime
...
IDXXXXXXXX|1|Agent|Hi, thank you for contacting Samsung Technical Support.
|Text|2013-10-01T00:02:47+00:00
...
```

### data/session_satisfaction.txt
```
SessionID|Satisfaction
...
IDXXXXXXXX|Very  Satisfied
...
```

## Dependency

* Python 2.7.13
* Keras 2.0.6
* Tensorflow 1.1.0
* scikit-learn 0.18.2
* gensim 2.2.0

