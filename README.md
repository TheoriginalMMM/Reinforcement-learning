# First step : setup the environnement
Firstly, you need to creat a python virtual environnement using this command for exemple:
```
python3 -m venv DQN
```
After you activate your environnement, you will notice that your command console will be prefixed with the environnement name, which is named (DQN) as per our example.

```
source DQN/bin/activate
```
Then you can install the requirement of the project, using the command bellow:
```
pip install -r requirements.txt
```

When you are finished in the DQN environnement, to exit type:

```
deactivate
```

# Hwo to train/test a agent ?

For the environnement CartPole-v1 you can train or/and test the agent by executing the file **CartPole-v1.py** using the command bellow:

```
python CartPole-v1.py
```

But before you have to open the file and set the boolean variables TRAIN TEST to choose what do you want to do, you can also choose the other paramters in the begining of this file.

For the environnement MineRL, tou can do this using the same method for the CartPole-v1 environnement, that's mean you have to modify the beginig of the file **MineRl.py** to set some parameters and other boolean variables to choose what do you want to do and then finaly run the command bellow.

```
python MineRl.py
```

By default runing the two last file will run one episode of each environnement using the pretrainned models.

# Credits:
- Mohamed El Mehdi MAKHLOUF
- Doriane AJMI 