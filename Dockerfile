FROM nvcr.io/nvidia/pytorch:25.03-py3
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

######## PIP
# OS 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    screen \
    tzdata \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 타임존을 기본값으로 설정 (UTC 예시)
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

WORKDIR /temp


# install all dependencies
COPY requirements.txt .
# Remove system-installed blinker package to avoid conflicts
RUN apt-get remove -y python3-blinker || true
RUN pip install -r requirements.txt

# WORKDIR /temp
# # (선택사항) Python 의존성이 명시된 파일이 있다면 복사 후 설치
# COPY lib_render ./lib_render/
# RUN pip install -e lib_render/simple-knn
# RUN pip install -e lib_render/diff-surfel-rasterization

# Install nvm, Node.js, and Claude Code
WORKDIR /temp
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    nvm install 24 && \
    nvm use 24 && \
    npm install -g @openai/codex

# Add nvm to bashrc for interactive shells
RUN echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc

CMD ["/bin/bash"]
