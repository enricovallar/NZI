## How to run this MPB based application?

At its core this is a Python Dash application that use MPB to do the calculations. 

In its current version MPB's Python interface is part of MEEP's Python interface. 
Hence, MEEP must be installed to make it work. 

MEEP is not available for ARM processors, so it cannot be installed on some Macbooks for example. 
If you have this problem you might want to use DTU HPC based on Alma Linux.
Otherwise you can just install it in your machine. 

## How to connect to DTU HPC
To connect to DTU HPC follow the guide on the offical website and use ThinLinc Client to get access.

When you are connected:

1. Click on Application
2. Click on Terminal Emulator. 
3. In the terminal write the command 'code'. It should open Visual Studio Code Vsc
4. (Skip to 6 if you are  already logged with your GitHub account)
    On the bottom-left corner click on Accounts and then "Backup and Sync Settings..."
5. You will be redirected to the GitHub login: follow the instructions. 
6. Fork [my repo](https://github.com/enricovallar/nzi-lithium-niobate) into your account. 
7. Go back to VSC and in a new window on the left panel click on Source Control (git symbol) and click on "Clone Repository". Clone the repository from GitHub following the instructions.

## Installing Miniconda
We need a virtual conda envirenment to install MEEP. 

Open a new 'Terminal'.
Write the following lines: 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/bin/activate

conda init
```


After this restart your terminal. 

## Installing MEEP
In the terminal:
```
conda create -n mp -c conda-forge pymeep pymeep-extras

conda activate mp
```
and reload the terminal


## Installing Dash
```
conda install dash

conda install dash-bootstrap-components

conda install dash-daq
```









