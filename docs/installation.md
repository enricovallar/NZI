## How to run this MPB based application?

!!! warning

    Linux is required. I use WSL with Ubuntu. But this can also work with HPC. 


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
3. In the terminal write the command 'code'.  It should open Visual Studio Code 

## Set up Visual Studio Code (VSC)

1. (Skip to 3 if you are  already logged with your GitHub account)
    On the bottom-left corner click on Accounts and then "Backup and Sync Settings..."
2. You will be redirected to the GitHub login: follow the instructions. 
3. Fork my repo [enricovallar/nzi-lithium-niobate](https://github.com/enricovallar/nzi-lithium-niobate) into your account.
4. Go back to VSC and in a new window on the left panel click on Source Control (git symbol) and click on "Clone Repository". Clone the repository from GitHub following the instructions.

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


## Activating the virtual envirenment
Now we install the most important dependences:

 - meep
 - dash
 - mkdocs


```bash 
conda create -n nzi-mp python=3.11 nzi-phc-finder -c enricovallar -c conda-forge -y
conda activate nzi-mp
```
Make sure to always use the python interpreter of this virtual envirenment


## Reading the docs
```bash 
mkdocs serve
```

## Running the application
When the right interpreter is chosen you just need to run ./src/app.py









