# SAVER
This repository contains the code to run the experiment (plus some additional experiments) described in the workshop paper “Quantile Regression Reinforcement Learning with State Aligned Vector Rewards” - https://openreview.net/forum?id=S1fuWRYCFm

## Requirements
    • Python v3.5
    • Tensorflow (https://www.tensorflow.org/install/)
    • OpenAI Gym (https://github.com/openai/gym)

## Installation
    • Download the folder and install requirements
    • Install and register the gym environments in the sub directory “saver_envs”
        ◦ $ pip install -e saver_envs
    • Run the file “main.py” to train an agent
        ◦ $ python main.py
