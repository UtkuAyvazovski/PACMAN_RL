# Reinforcement Learning for Ms. PACMAN

This project implements a reinforcement learning approach for training an agent to play Ms. PACMAN.

## Prerequisites

- Python 3.10.12
- Operating System: Ubuntu 22.04

## Setting up a Virtual Environment

1. Open a terminal.

2. Navigate to the project directory.

3. Create a virtual environment:
   ```bash
   python3.10 -m venv venv

## Activate virtual environment
Run the following line from the terminal:
   '''bash
   source venv/bin/activate
   
## Installing dependencies
Install the requirements from requirements.txt:
   '''bash
   pip install -r requirements.txt

## Run the simulation of the best model
Run the tryout.py, where the best model is defined in best.zip folder:
   '''bash
   python3 tryout.py

## Train the best model
Run the stabletrain.py and save the model in dqn_mspacman.zip folder:
   '''bash
   python3 stabletrain.py

# Windows 10 Approach

## Prerequisites

- Python 3.10
- Operating system: Windows 10

## Setting up a virtual environment

1. Download Python3.10 from Microsoft Store

2. Open Windows Powershell as the administrator

3. Navigate to the project directory.

4. Create a virtual environment:
   '''bash
   python3.10 -m venv venv

## Activate virtual environment
Run the following line in the Windows Powershell:
   '''bash
   .\venv\Scripts\activate

## Installing dependencies
Install the requirements from requirements.txt:
   '''bash
   pip install -r requirements.txt

## Run the simulation of the best model
Run the tryout.py, where the best model is defined in best.zip folder:
   '''bash
   python .\tryout.py

## Train the best model
Run the stabletrain.py and save the model in dqn_mspacman.zip folder:
   '''bash
   python .\stabletrain.py
