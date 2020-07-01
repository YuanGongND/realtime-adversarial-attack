
# Real-Time Adversarial Attacks
In recent years, many efforts have demonstrated that modern machine learning algorithms are vulnerable to adversarial attacks, where small, but carefully crafted, perturbations on the input can make them fail. While these attack methods are very effective, they only focus on scenarios where the target model takes static input, i.e., an attacker can observe the entire original sample and then add a perturbation at any point of the sample. These attack approaches are not applicable to situations where the target model takes streaming input, i.e., an attacker is only able to observe past data points and add perturbations to the remaining (unobserved) data points of the input. In this work, we propose a real-time adversarial attack scheme for machine learning models with streaming inputs.

## Cite us:  
If you feel this repository is helpful, please cite the following paper:

Yuan Gong, Boyang Li, Christian Poellabauer, and Yiyu Shi, **["Real-time Adversarial Attacks"](https://www.ijcai.org/Proceedings/2019/649)**, Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), Macao, China, August 2019.

## Dataset

In the experiments of this work, we use the [Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (2.3GB), which is publically accessible. You don't need to download it if you want to train the target model from scratch.

## How to run the code?
**1. Download the dataset and train the target model**
Clone this Github reporsitory. Then run:
```python
python src/speech_commands/train.py --data_dir data/
``` 
This will automatically download the Speech Commands dataset to the ``data`` directory and conduct the training. Note ``src/speech_commands/train.py`` is an official tutorial example of Tensorflow 1.0 but is discontinued by Google. Information about the model can be found in the comments in the head of the file. We use all default settings and the model should have around 90% accuracy. 

**2. Generate expert demonstration samples**
Under construction.

**3. Learn from the demonstration samples**


## Questions

If you have a question, please rasie an issue in this Github reporsity. You can also contact Yuan Gong (ygong1@nd.edu).
