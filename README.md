# Super Mario Project

## Group members 
- Benjamin Varian - 23215049
- Crystal Teh - 23209088

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

<!-- todo change to the file where the model gets loaded -->
To run the PPO model navigate into the agent folder, then agent again and run the following command.

```bash
    poetry run python3 <or python> model.py
```

This will load our final model up and will run through all the actions that it has saved.

The next agent is a deep q learning agent. To run this agent navigate into the agent folder, then modified_dqn_agent_v2 again and run the following command.

### TODO
-  CHANGE WHEN WE FINAL SUBMIT
-  update the actual file to run  

```bash
    poetry run python3 <or python> dqn.py
```

