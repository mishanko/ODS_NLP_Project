VERSION=24.02-py3
IMAGE_NAME=$(id -un)-project:${VERSION}
docker build --no-cache \
    --build-arg USER=$(id -un) \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -f Dockerfile -t ${IMAGE_NAME} .