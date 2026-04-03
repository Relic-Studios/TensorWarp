# TensorWarp Docker Image
# Multi-stage build: compile in builder, run in slim runtime

# Stage 1: Build
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .
RUN cargo build --release

# Stage 2: Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

COPY --from=builder /app/target/release/tensorwarp /usr/local/bin/
COPY --from=builder /app/vendor/cudarc /opt/tensorwarp/vendor/cudarc

# Python support
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install numpy

COPY python/ /opt/tensorwarp/python/
RUN pip3 install -e /opt/tensorwarp/python/

WORKDIR /workspace
ENTRYPOINT ["tensorwarp"]
CMD ["info"]
