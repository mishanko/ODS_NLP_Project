VERSION=24.02-py3
IMAGE_NAME=$(id -un)-project:${VERSION}
CONTAINER_NAME=$(id -un)-project
# Fix ENV variable
NCCL_P2P_LEVEL=NVL

docker run -d -it --rm \
    --shm-size 256G \
    --log-driver=none \
    --gpus all \
    --user ${USER}:${GROUP} \
    -e NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} \
    -v $(realpath /home/${USER}/.ssh):/home/${USER}/.ssh:ro \
    -v $(pwd)/..:/home/${USER}/ODS_NLP_Project \
    -v $(realpath /storage_research/${USER}/Projects/nlp-proj):/data \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}
