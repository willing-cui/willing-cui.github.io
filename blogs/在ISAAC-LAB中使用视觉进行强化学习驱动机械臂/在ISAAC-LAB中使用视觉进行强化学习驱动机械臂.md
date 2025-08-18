# 在ISAAC LAB中使用视觉进行强化学习驱动机械臂

## Direct和Manager的区别

Isaac Lab 提供两种工作流程——Direct和Manager。Direct 工作流程将为您提供最快速通往用于强化学习的工作自定义环境的路径，但 Manager based 工作流程将为您的项目提供更广泛开发所需的模块化。这意味着您可以设计您的项目以具有可根据不同需求替换的各种组件。例如，假设您想训练支持特定子集机器人的策略。您可以通过编写一个控制器接口层来设计环境和任务，以形式化我们其中一种 Manager 类。

除了 [`envs.ManagerBasedRLEnv`](https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv) 类之外，还可以使用配置类来为更模块化的环境提供支持， [`DirectRLEnv`](https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.html#isaaclab.envs.DirectRLEnv) 类允许在环境脚本化中进行更直接的控制。

直接工作流任务实现完全奖励和观察功能的直接任务脚本。这允许在方法的实现中更多地控制，比如使用Pytorch jit功能，并提供一个更少抽象的框架，更容易找到各种代码片段。

## 项目参考

这里我们主要使用两个项目进行参考

- Cartpole-RGB-ResNet18-v0 任务的注册代码于文件

  C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\__init__.py

  ```python
  gym.register(
      id="Isaac-Cartpole-RGB-ResNet18-v0",
      entry_point="isaaclab.envs:ManagerBasedRLEnv",
      disable_env_checker=True,
      kwargs={
          "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleResNet18CameraEnvCfg",
          "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
      },
  )
  ```

  对应上面的代码，任务环境注册于文件

  C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\cartpole_camera_env_cfg.py

  ```python
  @configclass
  class CartpoleRGBCameraSceneCfg(CartpoleSceneCfg):
  
      # add camera to the scene
      tiled_camera: TiledCameraCfg = TiledCameraCfg(
          prim_path="{ENV_REGEX_NS}/Camera",
          offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
          data_types=["rgb"],
          spawn=sim_utils.PinholeCameraCfg(
              focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
          ),
          width=100,
          height=100,
      )
      
      
  @configclass
  class CartpoleRGBCameraEnvCfg(CartpoleEnvCfg):
      """Configuration for the cartpole environment with RGB camera."""
  
      scene: CartpoleRGBCameraSceneCfg = CartpoleRGBCameraSceneCfg(num_envs=512, env_spacing=20)
      observations: RGBObservationsCfg = RGBObservationsCfg()
  
      def __post_init__(self):
          super().__post_init__()
          # remove ground as it obstructs the camera
          self.scene.ground = None
          # viewer settings
          self.viewer.eye = (7.0, 0.0, 2.5)
          self.viewer.lookat = (0.0, 0.0, 2.5)
          
          
  @configclass
  class ResNet18ObservationCfg:
      """Observation specifications for the MDP."""
  
      @configclass
      class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
          """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""
  
          image = ObsTerm(
              func=mdp.image_features,
              params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
          )
  
      policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()
      
      
  @configclass
  class CartpoleResNet18CameraEnvCfg(CartpoleRGBCameraEnvCfg):
      """Configuration for the cartpole environment with ResNet18 features as observations."""
  
      observations: ResNet18ObservationCfg = ResNet18ObservationCfg()
  
  ```

  这些继承文件环环相扣，有很深的继承关系。

## 操作步骤

1. 复制文件 C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env_cfg.py 到同一路径下，新文件修改名称为

   C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_camera_env_cfg.py.

2. 将新文件中 FactoryEnvCfg 类名替换为 FactoryCameraEnvCfg

3. 同理，复制文件 C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_env.py 为 C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\factory\factory_camera_env.py 并做类名替换。注意将import中对应修改cfg文件为新复制的camera_env_cfg.

   ```python
   from .factory_camera_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryCameraEnvCfg
   ```

4. 下面修改 factory_camera_env_cfg.py 文件，参照：C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\cartpole\cartpole_camera_env.py

   ```python
   from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file # import camera configurations
   ```

5. 想要让新编写的环境注册运行，需要在下面的文件中添加相关的描述

   C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\factory\__init__.py