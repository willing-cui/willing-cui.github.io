## 相关代码库及链接

1. <a href="https://github.com/isaac-sim/IsaacSim" target="_blank" rel="noopener noreferrer">NVIDIA Isaac Sim Github</a>

> NVIDIA Isaac Sim™ 是一个基于 NVIDIA Omniverse 的仿真平台，旨在用于在逼真的虚拟环境中开发、测试、训练和部署 AI 驱动的机器人。它支持从 URDF、MJCF 和 CAD 等常用格式导入机器人系统。该仿真器利用高保真、GPU 加速的物理引擎来模拟精确的动力学，并支持大规模的多传感器 RTX 渲染。它配备了端到端的工作流程，包括合成数据生成、强化学习、ROS 集成和数字孪生仿真。Isaac Sim 为机器人开发的各个阶段提供支持所需的基础设施。

2. <a href="https://github.com/isaac-sim/IsaacLab" target="_blank" rel="noopener noreferrer">NVIDIA Isaac Lab Github</a>

> **Isaac Lab**是一个基于 GPU 加速的开源框架，旨在统一和简化机器人研究工作流程，例如强化学习、模仿学习和运动规划。它基于[NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)构建，结合了快速、精确的物理和传感器仿真，使其成为机器人领域仿真到实际应用的理想选择。
>
> Isaac Lab 为开发者提供了一系列用于精确传感器仿真的关键功能，例如基于 RTX 的摄像头、激光雷达或接触式传感器。该框架的 GPU 加速功能使用户能够更快地运行复杂的仿真和计算，这对于强化学习和数据密集型任务等迭代过程至关重要。此外，Isaac Lab 既可在本地运行，也可在云端部署，为大规模部署提供了灵活性。
>
> 有关 Isaac Lab 的详细描述，请参阅我们的[arXiv 论文](https://arxiv.org/abs/2511.04831)。

## 安装过程

### 0. 安装依赖项

- [**Git**](https://git-scm.com/downloads)：用于版本控制和代码仓库管理
- [**Git LFS**](https://git-lfs.com/)：用于管理仓库中的大型文件
- **（仅限 Windows - C++）Microsoft Visual Studio（2019 或 2022） ：您可以从**[Visual Studio 下载中心](https://visualstudio.microsoft.com/downloads/)安装最新版本。请确保已选择“ **使用 C++ 的桌面开发”工作负载。** 
- **（仅限 Windows - C++）Windows SDK**：请将其与 MSVC 一起安装。您可以在 Visual Studio 安装程序中找到它。

<span class="image main">
<img class="main img-in-blog" style="max-width: 80%" src="./blogs/56_Install_Isaac_on_Windows/MSVS_1.webp" alt="img_name" />
<i>MSVS安装配置（1）</i>
</span> 

<span class="image main">
<img class="main img-in-blog" style="max-width: 80%" src="./blogs/56_Install_Isaac_on_Windows/MSVS_2.webp" alt="img_name" />
<i>MSVS安装配置（2）</i>
</span> 

### 1. 克隆代码库

```bash
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim
git lfs install
git lfs pull
```

- `git lfs install`是 Git LFS（Git Large File Storage，Git 大文件存储）的一个初始化命令。
  它的作用是在你的当前 Git 仓库中设置并启用 Git LFS 功能。具体来说：
  **安装钩子**： 它会在你的本地 Git 仓库的 .git/hooks目录下安装必要的“钩子”脚本。这些钩子使得当你执行 `git add`、`git commit`等操作时，Git LFS 能够自动拦截对大文件的操作，并将其替换为文本指针，而不是将庞大的二进制文件本身存入版本历史。
  **仓库级配置**： 此命令是针对单个 Git 仓库进行的配置。你需要在每个需要使用 Git LFS 管理大文件的仓库中运行一次。
  简单来说，这个命令是告诉你仓库：“从今以后，我要在这里用 LFS 来管理某些类型的大文件了。”

### 本地编译建造

鉴于本文写作时，MSVS已更新至2026版本，使用最新版本进行安装。需要对项目配置文件`isaacsim\repo.toml`做下列改动（[参考在线文档](https://github.com/isaac-sim/IsaacSim/blob/main/docs/readme/windows_developer_configuration.md)）

```bash
## 1） 删除对VS版本的指定
# Filter on Visual Studio version e.g.: Visual Studio 2022. Empty string will match all years and prioritize the newest.
vs_version = ""

## 2） 将vs_path指向新版本的安装位置
# Visual Studio path; This will be used if the user would like to point to a specific VS installation rather than rely on heuristic locating.
vs_path = "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\"

## 3） 将winsdk_path直向新版本的安装位置，需要注意的是，"C:\\Program Files (x86)\\Windows Kits\\10\\bin"下存放有多个编译版本，此处选择编号最大的（最新的）版本
# Windows SDK path; This will prevent needing to dynamically locate an installation by guesswork.
winsdk_path = "C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.26100.0"

## 4） 将"platform:windows-x86_64".enabled 标签设置为true，注意保持其他标签的值不变
[repo_docs]
"platform:linux-aarch64".enabled = false # Disable docs generation for aarch64 platforms, as it is not supported.
"platform:windows-x86_64".enabled = true # Disable docs generation for windows64 platforms, as it is not supported.
"platform:linux-x86_64".enabled = true

## 5） 由于vs2026改变了文件工程的后缀，需要将isaac-sim.sln改为isaac-sim.slnx
[repo_build.msbuild]
sln_file = "isaac-sim.slnx"
```

另外，需要从premake5的Github仓库下载最新的beta8版本，替换原有的`premake5.exe`。[下载链接](https://github.com/premake/premake-core/releases)

对于Windows系统，可在Git Bash命令行中（isaacsim路径下）执行脚本：

```bash
build.bat --skip-compiler-version-check
```

### 编译中所遇到的问题

1. `action`变量为空的问题

   ```bash
   Error: D:/packman-repo/chk/repo_build/1.16.0/lua/omni/repo/build/compilecommands.lua:87: attempt to index a nil value (local 'action')
   ```
	改写文件，加入判断语句，让空变量的情况直接return。

    ```lua
    local function execute()
        -- register our workload to run after all workspaces, projects and configs have been configured
        local action = premake.action.current()
        -- 添加空值检查
        if not action then
            print("Warning: action module not available")
            return
        end
        ...
    ```

2. `FatalCompileWarnings`标志位不再受支持的问题。

   新版本的premake5-beta8正式移除了该标志位，需要将涉及该标志位的代码段全部改写删除（可以根据报错内容定位需要修改的代码位置）。[参考链接](https://premake.github.io/docs/flags/)

3. Error: no such action 'vs18'

   上面的报错是由premake5-beta8产生。从premake5的Github仓库下载最新的beta8版本，替换原有的`premake5.exe`后，新版本的premake5支持的编译环境代号有：

   ```bash
   ACTIONS
   
    clean             Remove all binaries and generated files
    codelite          Generate CodeLite project files
    gmake             Generate GNU makefiles for POSIX, MinGW, and Cygwin
    gmakelegacy       Generate GNU makefiles for POSIX, MinGW, and Cygwin
    ninja             Generate Ninja build files
    vs2005            Generate Visual Studio 2005 project files
    vs2008            Generate Visual Studio 2008 project files
    vs2010            Generate Visual Studio 2010 project files
    vs2012            Generate Visual Studio 2012 project files
    vs2013            Generate Visual Studio 2013 project files
    vs2015            Generate Visual Studio 2015 project files
    vs2017            Generate Visual Studio 2017 project files
    vs2019            Generate Visual Studio 2019 project files
    vs2022            Generate Visual Studio 2022 project files
    vs2026            Generate Visual Studio 2026 project files
    xcode4            Generate Apple Xcode 4 project files
   ```

   需要在`isaacsim\_repo\deps\repo_build\omni\repo\build\main.py`中改写对应的配置，硬性写明编译环境版本

   ```python
   def build(
       repo_folders,
       platform_host,
       platform_target,
       configs: List[str],
       settings: Settings,
       verbose: bool = False,
       jobs: Union[int, str] = -1,
       extra_args: List = [],
       build_target=None,
       keep_going: bool = False,
   ):
       """Cross platform build command wrapper."""
       
       settings.vs_version="vs2026"
       ...
   ```

4. 无法找到sln工程文件的问题，参考上一节第5条：可能由于vs2026改变了文件工程的后缀，需要将isaac-sim.sln改为isaac-sim.slnx

   ```bash
   BuildError: Windows iterative build with repo_build.msbuild.vs_version unset failed to find the expected solution file symlink D:\isaacsim\_compiler\vs2026\current\isaac-sim.sln.
   ```

## 安装成功

这篇博客记录了在VS2026环境下安装IsaacSim的过程，尽管IsaacSim当前（2026年3月29日）并没有增加对VS2026的支持，通过修改工程代码，在经历些许波折之后仍可以安装成功。PS：MSVS旧版本需要付费订阅才能下载。

或许在将来不久，IsaacSim便会提供对新版VS的支持。

<span class="image main">
<img class="main img-in-blog" style="max-width: 100%" src="./blogs/56_Install_Isaac_on_Windows/Install_Succeeded.webp" alt="img_name" />
<i>安装成功</i>
</span> 