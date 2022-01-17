# Installation
1. Soyez sûr d'avoir `JDK 1.8` d'installé.

    *Pour vérifier si vous avez la bonne version* :
    ```bash
    java -version  # Attendu : java version '1.8.x_xxx' ...
    ```

    Pour l'installer avec *sudo* :
    ```bash
    sudo add-apt-repository ppa:openjdk-r/ppa
    sudo apt-get update
    sudo apt-get install openjdk-8-jdk
    ```

    Pour l'installer **sans** *sudo* :
    ```bash
    wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie" https://javadl.oracle.com/webapps/download/GetFile/1.8.0_281-b09/89d678f2be164786b292527658ca1605/linux-i586/jdk-8u281-linux-x64.tar.gz
    
    tar -xvzf jdk-8u281-linux-x64.tar.gz jdk1.8.0_281/

    # À faire à chaque fois que vous ouvrez un nouveau terminal :
    # (Si vous ne voulez pas le refaire à chaque fois, mettez la ligne dans votre .bashrc)
    export PATH=$PWD/jdk1.8.0_281/bin:$PATH
    ```

2. Installez le paquet `minerl`.
    
    À faire en prérequis **sur les ordinateurs de la fac'** :
    ```bash
    pip3 uninstall opencv-python
    pip3 install opencv-python==4.4.0.44
    pip3 install wheel
    ```

    ```bash
    pip3 install --upgrade minerl  # Peut demander de faire `sudo apt install libpython3.x-dev` sur vos machines perso'
    ```

3. Ajoutez les environnements customisés.
    - Téléchargez `custom_environments.zip` sur Claroline.
    - Décompressez l'archive à la racine de vos sources (là où se trouve votre fichier principal, e.g. `main.py`)
    - Dans ce fichier principal, importez juste le paquet pour ajouter les environnements au registre Gym :
        ```python
        import custom_environments
        ```
