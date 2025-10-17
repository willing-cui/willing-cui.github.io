# 在ISAAC LAB中使用视觉进行强化学习驱动机械臂

## Direct和Manager的区别

Isaac Lab 提供两种工作流程——Direct和Manager。Direct 工作流程将为您提供最快速通往用于强化学习的工作自定义环境的路径，但 Manager based 工作流程将为您的项目提供更广泛开发所需的模块化。这意味着您可以设计您的项目以具有可根据不同需求替换的各种组件。例如，假设您想训练支持特定子集机器人的策略。您可以通过编写一个控制器接口层来设计环境和任务，以形式化我们其中一种 Manager 类。

除了 [`envs.ManagerBasedRLEnv`](https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv) 类之外，还可以使用配置类来为更模块化的环境提供支持， [`DirectRLEnv`](https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.html#isaaclab.envs.DirectRLEnv) 类允许在环境脚本化中进行更直接的控制。

直接工作流任务实现完全奖励和观察功能的直接任务脚本。这允许在方法的实现中更多地控制，比如使用Pytorch jit功能，并提供一个更少抽象的框架，更容易找到各种代码片段。