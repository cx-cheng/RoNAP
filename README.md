# 一个通用机器人导航分析平台（RoNAP）

## 简介

机器人导航任务是在大量静止或移动障碍物中找到一条无碰撞路径。各种成熟的算法已被用于解决导航任务。有必要在实践中测试所设计的导航算法的性能。
然而，直接在真实环境中实施这些算法似乎是一个极其不明智的选择，除非能保证其性能是可接受的。否则，由于训练过程漫长，测试导航算法需要花费大量时间，而不完美的性能可能会在机器人与障碍物碰撞时造成严重的机器损坏。
因此，开发一个模拟真实环境的移动机器人分析平台具有重要意义，该平台能够准确复制应用场景，操作简单。
本文介绍了一个全新的分析平台，名为机器人导航分析平台（RoNAP），这是一个基于 Python 环境开发的开源平台。它具有友好的用户界面，支持各种导航算法的实现和评估。
各种现有算法在该平台上都取得了理想的测试结果，这表明该平台在导航算法分析方面具有可行性和高效性。

## 安装
* 操作系统： Windows7/windows10, 32/64位
* Python 3.6 or above
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Pygame](www.pygame.org)

## 用法
* 使用任意Python编译器（Pycharm、Spyder、IDLE……）运行主程序，进入初始界面，界面中有六个选择按钮
  <p align="center">
    <img src="/interface/welcome.PNG" width="500px">
  </p>
* 点击【Start】按钮进入算法导航模块，该模块主要用于训练和测试算法的导航性能
  <p align="center">
    <img src="/interface/start.PNG" width="500px">
  </p>
* 点击【Collect】按钮进入人工导航模块，该模块主要用于收集人类行为数据，为深度学习算法训练提供标签数据
* 在系统初始界面点击【Maze Generator】，可以进行地图场景更新，如图2.4所示，点击【Load Map】，下一次机器人将在新的场景下运动
  <p align="center">
    <img src="/interface/maze_load.PNG" width="500px">
  </p>
* 在系统初始界面点击【Configurations】，进入User Configuration（用户配置）界面
  <p align="center">
    <img src="/interface/configuration.PNG" width="500px">
  </p>
* 在系统初始界面点击【Save Data】可以保存当前回合机器人在每个时间步下的移动数据

## 引用
```
@Article{s22239036,
  AUTHOR = {Cheng, Chuanxin and Duan, Shuang and He, Haidong and Li, Xinlin and Chen, Yiyang},
  TITLE = {A Generalized Robot Navigation Analysis Platform (RoNAP) with Visual Results Using Multiple Navigation Algorithms},
  JOURNAL = {Sensors},
  VOLUME = {22},
  YEAR = {2022},
  NUMBER = {23},
  ARTICLE-NUMBER = {9036},
  URL = {https://www.mdpi.com/1424-8220/22/23/9036},
  PubMedID = {36501739},
  ISSN = {1424-8220},
  DOI = {10.3390/s22239036}
}
```
