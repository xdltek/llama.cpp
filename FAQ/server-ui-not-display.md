# llama-server Web UI 无法显示

## 1. 问题描述

在部分运行环境中，启动 `llama-server` 后：

- llama-server 服务可以正常启动；
- API 接口可以正常访问；
- 模型推理功能正常；
- 但是通过浏览器访问 Web UI 页面时无法正常显示 llama.cpp Server UI。


---

## 2. 问题原因

llama.cpp server 模式包含 Web UI 前端资源。

部分机器环境由于：

- Node.js 环境缺失；
- Node.js 版本不兼容；
- 编译过程中未生成完整 Web UI 静态文件；

导致 `llama-server` 后端正常运行，但是 Web UI 页面无法显示。


---

## 3. 解决方案

确认 Node.js 版本：

```
node -v
```

如果未安装 Node.js 22，根据一下步骤修复

#### 3.1 安装 Node.js 22

使用 nvm 管理 Node.js。

安装 nvm

```bash
git clone https://gitee.com/mirrors/nvm.git ~/.nvm

cd ~/.nvm

git checkout v0.40.1
```

加载 nvm：

```
source ~/.nvm/nvm.sh
```

配置环境变量：

```
echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc

echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc
```

配置国内 Node.js 镜像：

```
export NVM_NODEJS_ORG_MIRROR=https://npmmirror.com/mirrors/node
```

安装 Node.js 22：

```
nvm install 22
```

确认版本：

```
node -v
```

#### 3.2 重新编译运行 llama.cpp

参考README.md → Quick Start 