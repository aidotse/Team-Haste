On AZ server:

### 1. Build the docker image if it doesn't exist
`docker build -t 3h4m/haste .`

### 2. Run training with docker
`./run_with_docker.sh -c config.json`

### 3. Run the shell with docker
`docker run --shm-size=24gb -it --rm -v /home/group5/:/workspace 3h4m/haste bash`

---

### Install docker
`curl -sSL https://get.docker.com/ | sh`

### Pull the image 
`sudo docker pull 3h4m/haste`

### Connect to the kernel
`sudo docker run -it 3h4m/haste bash`
