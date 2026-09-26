FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
ARG KENLM_COMMIT=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev \
    ca-certificates curl time fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 3 https://codeload.github.com/kpu/kenlm/tar.gz/${KENLM_COMMIT} -o /tmp/kenlm.tgz \
    && mkdir /opt/kenlm && tar -xzf /tmp/kenlm.tgz --strip-components=1 -C /opt/kenlm \
    && cmake -S /opt/kenlm -B /opt/kenlm/build -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=6 \
    && cmake --build /opt/kenlm/build -j 4 \
    && rm /tmp/kenlm.tgz
RUN pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 Cython==3.1.2 \
    && cd /opt/kenlm && pip install --no-cache-dir --no-build-isolation . \
    && pip install --no-cache-dir numpy==2.2.6 matplotlib==3.10.3 psutil==7.0.0 pytest==8.4.1 nltk==3.9.2
RUN printf '%s\n' "${KENLM_COMMIT}" > /opt/kenlm/COMMIT
ENV PATH="/opt/kenlm/build/bin:${PATH}" PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /workspace
CMD ["python", "-m", "ngram_lab.kenlm_cli", "--help"]
