
# Real-Time Adversarial Attacks
In recent years, many efforts have demonstrated that modern machine learning algorithms are vulnerable to adversarial attacks, where small, but carefully crafted, perturbations on the input can make them fail. While these attack methods are very effective, they only focus on scenarios where the target model takes static input, i.e., an attacker can observe the entire original sample and then add a perturbation at any point of the sample. These attack approaches are not applicable to situations where the target model takes streaming input, i.e., an attacker is only able to observe past data points and add perturbations to the remaining (unobserved) data points of the input. In this work, we propose a real-time adversarial attack scheme for machine learning models with streaming inputs.

## Cite us:  
If you feel this repository is helpful, please cite the following paper:

Yuan Gong, Boyang Li, Christian Poellabauer, and Yiyu Shi, **["Real-time Adversarial Attacks"](https://www.ijcai.org/Proceedings/2019/649)**, Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), Macao, China, August 2019.

## Dependencies
Tensorflow 

## Dataset

In the experiments of this work, we use the [Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (2.3GB), which is publically accessible. You don't need to download it if you want to train the target model from scratch.

## How to run the code?
**1. Download the dataset and train the target model**

Clone this Github reporsitory. Then run:
```python
python src/speech_model_train/speech_commands/train.py --data_dir data/
``` 
This will automatically download the Speech Commands dataset to the ``data`` directory and conduct the training. Note ``src/speech_model_train/train.py`` is an official tutorial example of Tensorflow 1.0 but is discontinued by Google. Information about the model can be found in the comments in the head of the file. We use all default settings and the model should have around 90% accuracy. 

The default saved model can only infer one sample at a time. We recommend to modify it by using the provided ``src/speech_model_train/freeze_batch.py`` script to allow the saved model conduct batch inference. You need to: 
1) Change line 86 of ``src/speech_model_train/freeze_batch.py``to your desired inference batch size (we use 50); 
2) Run ``python src/speech_model_train/freeze_batch.py --start_checkpoint=/tmp/speech_commands_train/conv.ckpt-xxxx --output_file=/tmp/my_frozen_graph.pb
``

This will significantly speed up the next step of generating expert demostration samples. We include some pre-trained model with different inference batch size in ``src/speech_model_train/trained_models/``.

**2. Generate expert demonstration samples (i.e., non-real-time adversarial samples)**

 As mentioned in the paper, we generate the expert demonstration samples using a non-real-time adversarial example generation method, specifically, we use an audio version of the ["one-pixel attack"](https://arxiv.org/abs/1710.08864) (Jiawei Su, Vasconcellos Vargas, Sakurai Kouichi, IEEE Transactions on Evolutionary Computation, 2019). 

The implementation of this is in ``src/generate_expert_demo.py``.  In line 242, the number of perturbed "pixels" (in the audio context, the number of purturbed segments) is defined, in this work, we use 5. In line 268, one sample is feeded to the attack model, where ``generate_perturbation_fix_scale`` is the key function, in which ``attack_fix_scale`` is the key function, in which ``differential_evolution `` is the key function. For ``differential_evolution ``, two important parameters are ``maxiter`` and ``popsize``. Higher values for both do lead to better attack performance, but also increase the computation overhead, in this work, we use 5 and 10, respectively (note here there is a typo in the paper that we said the maxiter is 75). 

To run the code, first change the path of the target model and dataset in line 247 and line 251 and then run:

```python
python src/generate_expert_demo.py
```
The perturbed audios will be saved in the loacation specified in line 212, and the log will be saved in the location specified in line 274.

Please note that since we needs to generate more than 10,000 samples, each involves an iterative evolution optimization algorithm, it can take a very long time (a few days) to finish. Parallelization can significantly accelerates the process, you need to first 1) modify and save a target model that allows batch inference; 2) use the ``src/differential_evolution `` module (authored by Dan Kondratyuk, 2018) instead of ``scipy.optimize.differential_evolution``. The ``src/generate_expert_demo.py`` script has the parallelization implementation included, where we use the batch inference size of 50 (defined in line 42).

**3. Learn from the demonstration samples**


**4. Analyze the attack result**


## Questions

If you have a question, please rasie an issue in this Github reporsity. You can also contact Yuan Gong (ygong1@nd.edu).
