# Lenta Hack
This repository contains the machine learning part of the solution by our awesome team Pivo-T!

## Model
The model inside looks as follows:

<a href="https://ibb.co/wLjDQDP"><img src="https://i.ibb.co/1njHXHP/untitled-2x.png" alt="untitled-2x" border="0"></a>

Generally speaking, it aggregates all the temporal info we have about user's transactions via article embeddings and LSTM, and then classifies the user as a "leaving" or not. We used Pytorch Lightning as a supportive framework to develop the model.

## Target & evaluation

We define a "leaving" client as a person, who buys in the next months goods on a total cost less than the minimal among the previous three months. For each client and for a certain timestamp we define a classification task on whether he will become "leaving" in the next month.

One of the major problems of this setting is a high class imbalance (leaving clients add up to ~6-7% of total sample), which leads to unstable model performance and threshold uninterpretability. While experimenting we maximized ROC-AUC as it is a non-symmetric threshold-free metric, and it allows to focus on the minorly represented class.

We've validated the model carefully on the future data that we weren't training on. The procedure resembles standart time series cross-validation. We take several months, fit to predict the probability of leave in the next one, and then move one month forward again to validate the model.

## Experiments
Some of our experiments are presented in W&B:
https://wandb.ai/waytobehigh/lenta_hackathon?workspace=user-waytobehigh
We cleansed out all the debugging and failed stuff so that you could enjoy pretty plots :)

## Docker setup
#### If you don't want to run training in docker, you can proceed to packages section
Dockerfile describes Nvidia Ubuntu 18.04 docker image with python3.8 and some basic packages installation.

### Optional:
You may want to build and run docker container in a tmux terminal, to detach from it and connect later:
```
sudo apt update && apt upgrade
apt install -y tmux
tmux new -s <some_name>
```
 
### Build and run
Run following commands to build docker image and then run a container:
```
. docker-build.sh
. docker-run.sh
```

If you used tmux, you can now detach with ctrl+b d and create further connections to the container via
```
docker exec -it <container_hash> /bin/bash
```

## Environment and packages
If you used docker build, then you can enable created python environment from inside of the container with
```
. ../.env/bin/activate
```

Otherwise, you need to create new virtual environment with your favourite tool and install dependencies, e.g.
```
virtualenv .env --python=python3.8
. .env/bin/activate
pip install -r requirements.txt
```

## Use the model
```
python train.py
```
