FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt-get update -y && apt-get install rustc cargo -y
RUN python -m pip install uv && BLIS_ARCH=generic uv pip install --break-system-packages --system -e "/data-pipeline-pretrain"
RUN uv pip install --break-system-packages --system ninja && MAX_JOBS=8 uv pip install --break-system-packages --system --no-build-isolation --no-deps -U git+https://github.com/facebookresearch/xformers.git@v0.0.26.post1#egg=xformers