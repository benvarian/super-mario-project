# Super Mario Project

## Group members 
- Benjamin Varian - 23215049, [Email Ben](mailto:23215049@student.uwa.edu.au)
- Crystal Teh - 23209088, [Email Crystal](mailto:23209088@student.uwa.edu.au)

## Project Overview 
In this project, you will develop AI agents to control the iconic character Mario in the classic game Super Mario Bros using the gym-super-mario-bros environment. The main objective is to implement at least two distinct AI algorithms/methods and compare their performance, strengths, and weaknesses in the context of playing the game.

## How to run 
For this project we picked poetry as its a bit easier to run than conda. If you dont already have poetry installed please do so by following this link [poetry](https://python-poetry.org/docs/). 

Just a note for windows users, if you do plan on running this project please use [chocolatey](https://chocolatey.org/) to install poetry as i think it is easier.

Once you have poetry installed you can run the following commands to get the project up and running. 

``` bash
    cd agent && poetry install
```

Now that a virtual environment has been created and all the dependencies have been installed you can run the following command to run the project. 

## PPO Agent
**It is worth noting that with the gym version of anything less than 0.26.0 means this agent wont run, so if there is any errors please test with the version of gym been > 0.26.0.** To update the package please run:

```bash 
    poetry add gym==0.26.0
```

For the next file to work, please make sure that you have downloaded our model from [here](https://uniwa-my.sharepoint.com/:u:/g/personal/23215049_student_uwa_edu_au/Ecg7bZjBptNGkPNCi4KZ-L8BQ6uSE7T252DRSeIkm-e46A?e=30OPvC) or [here](https://drive.google.com/file/d/1Tk-wObTZvUFX92BJvWBmJLneVnSg_3zw/view?usp=sharing) and place it in the train folder located inside agent. To run the PPO model navigate into the agent folder, then agent again and run the following command.
```bash
    poetry run python3 analyse.py
```

If there are any problems with the downloading process of our model please reach out as we understand the importance of running this file.

## DQN Agents
**Please note: to run this DQN agent we need a different gym version from what we used in the PPO agent. It has been tested thoroughly on version 0.23.1.** To update please run:
``` bash
    poetry add gym==0.23.1
```
The next agent is the DQN agent. To run this agent navigate back into the agent folder, then DQN Agents and run the following command.

```bash
    poetry run python3 run_mario.py
```
This will firstly run through the model that has been trained and perform the actions that it has saved.

## Rule Based Agents

**Please note: to run these rule based agents need a different gym version from what we used in the DQN agent. It has been tested thoroughly on version 0.26.0.** To update please run:
``` bash
    poetry add gym==0.26.0
```

To run our rule based agents we will be demoing the v2 and v3 versions. To run these please run the following commands:

```bash
    poetry run python3 rule_based_v2.py
    poetry run python3 rule_based_v3.py
```

## Problems 

If there are any problems with versioning and applying api compatability, please make sure that you have the correct version of gym installed.

If issues persist please use the links at the top of the readme to contact us.