# raspi4B_scenes_demo
基于树莓派4B与Paddle-Lite实现的场景分类(5分类)


## 环境要求

* ARMLinux
    armLinux即可，64位与32位系统都可运行，[Paddle-Lite预编译库](https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html)
    
    * gcc g++ opencv cmake的安装（以下所有命令均在设备上操作）
    ```bash
    $ sudo apt-get update
    $ sudo apt-get install gcc g++ make wget unzip libopencv-dev pkg-config
    $ wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
    $ tar -zxvf cmake-3.10.3.tar.gz
    $ cd cmake-3.10.3
    $ ./configure
    $ make
    $ sudo make install
    ```
## 安装
$ git clone https://github.com/hang245141253/raspi4B_scenes_demo

## 目录介绍

scenes文件夹下为项目源码

Paddle-Lite文件夹为Paddle-Lite的预测库，包含32位与64位的预测库。库版本是Paddle-LiteV2.6.0。

## 使用

进入scenes文件夹，提供两个脚本。cmake.sh用于编译程序，run.sh用于预测。

执行sh cmake.sh编译代码。

然后执行run.sh预测五张场景图像。

以下是run.sh脚本的部分代码：

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./scenes ../models/scenes.nb ../images/desert.jpg ../label
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./scenes ../models/scenes.nb ../images/church.jpg ../label
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./scenes ../models/scenes.nb ../images/river.jpg ../label
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./scenes ../models/scenes.nb ../images/ice.jpg ../label
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./scenes ../models/scenes.nb ../images/lawn.jpg ../label
```

  程序会运行5次，按键盘上的“0”或者空格即可停止运行程序（注意按“0"之前需要点击一下跳出来的图片结果预测框）
  
  项目默认环境是armlinux 64位。如果您的系统是armlinux32位的，需要自行在cmake.sh与 run.sh中将TARGET_ARCH_ABI=armv8 注释掉，并取消#TARGET_ARCH_ABI=armv7hf的注释即可。
