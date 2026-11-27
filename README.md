# Newton_test_tools

## 1. 工具说明

`newton-test` 统一运行参考引擎和参评引擎。Python 适配器用已有 example 初始化场景，并提供步进、同步和读数接口。公用工具加载适配器、执行精度和性能流程、计算误差、汇总 FPS 和峰值显存，以及复核保存的结果。两端可以使用各自的运行环境和编程语言。

任务 1—3 覆盖 11 个物理场景；流固耦合的两个版本分别运行，共 12 组。下表列出[常规清单](cases/regular6.json)和[扩展清单](cases/extended.json)中的运行参数：

| 文档场景 | 场景 ID | 求解器 | world 数 | 每帧子步数 |
| --- | --- | --- | ---: | ---: |
| 1-1 基本形状 | `basic_shapes` | XPBD | 1 | 5 |
| 1-2 传送带 | `basic_conveyor` | XPBD | 1 | 10 |
| 1-3 螺母与螺栓 | `nut_bolt_hydro` | XPBD | 20 | 4 |
| 1-4 十万自由度刚体 | `basic_shapes_100k` | XPBD | 49 | 5 |
| 1-5 摆 | `basic_pendulum` | XPBD | 1 | 10 |
| 1-6 人形机器人 | `robot_g1` | MuJoCo 外部碰撞 | 4 | 6 |
| 1-7 机器狗行走 | `robot_anymal_c_walk` | MuJoCo 外部碰撞 | 1 | 4 |
| 1-8 十万自由度机器狗群组 | `anymal_c_walk_100k` | MuJoCo 外部碰撞 | 556 | 4 |
| 1-9 流体溃坝 | `implicit_mpm_dam_break` | 隐式 MPM | 1 | 2 |
| 1-10 流固耦合 V1 | `implicit_mpm_water_dam_single_box` | 隐式 MPM | 1 | 1 |
| 1-10 流固耦合 V2 | `fluid_grid_mpm_single_box` | MLS-MPM | 1 | 1 |
| 1-11 TG 涡 | `fluid_grid_taylor_green` | Stable-ACM | 1 | 6 |

求解器实现名称和边界类型以各引擎实际运行的场景清单为准。

| 场景 | 模式 | 确定性 | 场景级 graph | 每例每端默认运行 |
| --- | --- | --- | --- | --- |
| 常规六例 | 精度 | 开 | 关 | 500 帧，1 次 |
| 常规六例 | 性能 | 关 | 开 | 预热 5 帧后计时 500 帧，3 次 |
| 十万自由度刚体 | 精度 | 关 | 关 | 500 帧，3 次，各端指标取均值 |
| 十万自由度机器狗群组 | 精度 | 关 | 关 | 500 帧，3 次，各端指标取均值 |
| 两个十万自由度场景 | 性能 | 关 | 开 | 预热 5 帧后计时 500 帧，3 次 |
| 纯溃坝 | 精度 / 性能 | 关 | 关 | 精度 600 帧，1 次；性能预热 5 帧后计时 600 帧，3 次 |
| 流固耦合 V1、V2 | 精度 | 开 | 关 | 各 600 帧，1 次 |
| 流固耦合 V1、V2 | 性能 | 关 | 关 | 各预热 5 帧后计时 600 帧，3 次 |
| TG 涡 | 精度 / 性能 | 关 | 关 | 精度 2000 帧，1 次；性能预热 5 帧后计时 2000 帧，3 次 |

精度按完整轨迹计算碰撞形状对误差 `x_C`、动力学响应误差 `x_D` 和运动学误差 `x_K`，总误差 `x` 取三项的最大值。`x_K` 包含质心位置、姿态测地角、质心线速度和角速度。位置及速度使用清单的 `kinematic_scales`，姿态使用测地角除以 π。可用 `--repetitions` 单独诊断重复性。[计算口径](docs/adapter.md#精度数据)

性能在同一张 GPU 上顺序运行，每 5 帧同步一次。运行器用 NVML 每 10 ms 采样进程显存，汇总正式计时窗口内的驻留显存。FPS 为三轮总帧数除以总耗时，进程显存取三轮峰值的算术平均。两个比值分别用参评端结果除以参考端结果。

[扩展清单](cases/extended.json)含两个十万自由度场景、纯溃坝、流固耦合及 TG 涡。流固耦合的两种实现各跑一组，共用 Newton 隐式 MPM 的参考轨迹与性能记录。常规和扩展合计 11 个场景、12 组测试。

[任务 1—3 评分标准](docs/scoring.md)给出指标、计算公式与分值。Newton 场景脚本及资产位于 [场景包](examples/newton-scenes.zip)，解压后可查看和运行。

## 2. Newton 自检

以下命令使用 Linux、NVIDIA GPU、Git 和 `uv`。从空目录检出本工具和官方 Newton 1.6.0。工具与 Newton 参考适配器共用工具目录下的 `.venv`。

```sh
git clone https://github.com/FYTalon/Newton_test_tools.git
cd Newton_test_tools

NEWTON_SRC="$PWD/newton-v1.6.0"
git clone --depth 1 --branch v1.6.0 https://github.com/newton-physics/newton.git "$NEWTON_SRC"
uv sync --locked --python 3.11
uv pip install --python .venv/bin/python \
  -e "$NEWTON_SRC" -r adapters/newton/requirements.txt
```

先用 Newton 对 Newton 跑一次 `basic_shapes`。首次运行会按 Newton example 的配置获取资产。

```sh
NEWTON_ADAPTER=".venv/bin/python -m newton_test_tools worker --adapter adapters/newton/adapter.py"

uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$NEWTON_ADAPTER" \
  --repetitions 1 --performance-repetitions 1 \
  --output runs/newton-smoke

uv run --no-sync newton-test report runs/newton-smoke
uv run --no-sync newton-test verify runs/newton-smoke
```

自检会生成两端轨迹、计时和显存记录。`verify` 会检查文件哈希并重算报告。Newton 版本、资产来源和 Hydro 补丁说明见[参考适配器文档](adapters/newton/README.md)。

## 3. 写适配器（adapter）

参评双方都应公开实际用于测试的适配器源码、场景运行参数和标识映射，便于按同一套实现复验。每个引擎的适配器放在 `adapters/<引擎名>/`。现有入口：[Newton](adapters/newton/adapter.py) · [WanPhys](adapters/wanphys/adapter.py) · [NovaPhy](adapters/novaphy/adapter.py)。

Python 适配器实现 `prepare_trial(request, *, output)`，进入上下文时初始化场景并交出一个 `Trial`，退出时释放资源。可复制模板并接入自己的场景工厂。工具的 `worker` 读取请求、导入适配器并运行 `Trial`。[Python 适配器模板](adapters/example/adapter.py)

用请求里的 `case.id` 选择已有 example，按 `case` 读取求解器和物理配置。在构造场景前应用 `protocol.deterministic` 和 `protocol.graph`，并按 `gpu_devices` 选择设备。精度模式下，用 `reference_identity` 将本地刚体和形状编号映射到 Newton 的 `(world, name)` 标识。

| `Trial` 字段 | 提供的能力 |
| --- | --- |
| `engine`、`runtime` | 引擎显示名，以及实际帧间隔、设备 UUID、求解器、场景规模和两个开关。 |
| `step()` | 推进完整一帧及全部物理子步。 |
| `synchronize()` | 等待 GPU 完成仿真。 |
| `read_initial()` | 精度模式下提供模型、形状表、第 0 帧状态和速度约定。 |
| `observe()` | 精度模式下返回帧末的 `(bodies, contact_pairs)`。 |

精度由工具记录第 0 帧，再逐帧采集刚体状态和有效形状对。性能由工具预热、计时和生成报告，计时段调用步进和同步。其他语言可提供自己的可执行入口，接收 `--request` 和 `--output`，按协议写 `report.json` 和精度模式的 `trajectory.json`。模型、接触、计时及输出字段见[适配器协议](docs/adapter.md)。

## 4. 接入自己的引擎测试

在本工具根目录把候选命令指向自己的引擎。将下面的 Python 路径和适配器模块名换成自己的。模块须能从该环境导入，`--adapter` 也接受 Python 文件路径。

```sh
ENGINE_PY=/path/to/engine/.venv/bin/python
uv pip install --python "$ENGINE_PY" -e .
CANDIDATE_ADAPTER="'$ENGINE_PY' -m newton_test_tools worker --adapter your_engine.benchmark_adapter"
NEWTON_ADAPTER=".venv/bin/python -m newton_test_tools worker --adapter adapters/newton/adapter.py"
```

已有独立入口时，可将 `CANDIDATE_ADAPTER` 设为其完整命令，例如 `/path/to/engine/bin/benchmark_adapter`。运行器会追加本次试验的请求和输出参数。先检查适配器的状态和形状映射，再运行完整评测清单。

候选端的 Python 环境、适配器入口和场景清单按对应引擎的接入说明配置：[WanPhys](adapters/wanphys/README.md) · [NovaPhy](adapters/novaphy/README.md)。

```sh
uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --repetitions 1 --performance-repetitions 1 \
  --output runs/candidate-smoke

uv run --no-sync newton-test verify runs/candidate-smoke
```

首次运行含 `nut_bolt_hydro` 的正式精度前，对上面检出的 Newton 应用随附补丁。正式命令使用清单规定的帧数和重复次数。[Hydro 补丁说明](adapters/newton/README.md#20-world-hydro-的确定性参考)

```sh
git -C newton-v1.6.0 apply --check "$PWD/adapters/newton/hydro-v1.6.0.patch"
git -C newton-v1.6.0 apply "$PWD/adapters/newton/hydro-v1.6.0.patch"

uv run --no-sync newton-test run --suite cases/regular6.json --mode all \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --output runs/regular6

uv run --no-sync newton-test report runs/regular6
uv run --no-sync newton-test verify runs/regular6
```

每次指定新的输出目录。可重复传 `--case` 选择多个场景，可用 `--mode accuracy` 或 `--mode performance` 单独运行。默认监测 NVML 物理设备 0，`--gpu` 可传索引或 GPU UUID。Newton 适配器使用 `cuda:0`。有多张卡时，可用 `CUDA_VISIBLE_DEVICES` 让所选 GPU 成为两端进程内的 `cuda:0`。

常规六例默认每轮计时 500 帧，可用 `--performance-frames` 覆盖帧数。使用支持时长计时的适配器时，也可改用 `--duration`；仅支持固定帧数的适配器使用 `--performance-frames`。

输出目录的 `report.md` 汇总结果，`report.json` 保留逐帧误差和性能明细，`run.json` 记录实际测试配置。两端的请求、轨迹、显存采样和日志可在 `cases/<场景>/<模式>/` 下查到。

可以用已有轨迹离线复算精度：

```sh
uv run --no-sync newton-test compare --suite cases/regular6.json --case basic_shapes \
  --reference runs/newton-smoke/cases/basic_shapes/accuracy/reference/run-1/trajectory.json \
  --candidate runs/newton-smoke/cases/basic_shapes/accuracy/candidate/run-1/trajectory.json
```

### 运行扩展场景

扩展套件走同一个 `newton-test run` 和适配器入口。参考引擎环境须装有封存的扩展场景文件。[参考适配器](adapters/newton/README.md)

下面的命令运行两个流固耦合版本。

```sh
uv run --no-sync newton-test run --suite cases/extended.json \
  --case implicit_mpm_water_dam_single_box --case fluid_grid_mpm_single_box \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --output runs/coupled-mpm
uv run --no-sync newton-test verify runs/coupled-mpm
```

| 场景 ID | 正式精度采集 | 输出指标 |
| --- | --- | --- |
| `basic_shapes_100k` | 500 帧 × 5 子步，3 次 | 全程最大归一化总线动量变化和机械能偏差、实际穿透点平均深度、负法向力峰值及自由度；各端三次指标取均值后计算比值 |
| `anymal_c_walk_100k` | 500 帧 × 4 子步，3 次 | 同样四项全程指标，各端三次指标取均值后计算比值 |
| `implicit_mpm_dam_break` | 初态及 600 帧，1 次 | `x_F` 取逐粒子、逐帧的位置与线速度归一化 L2 误差最大值 |
| `implicit_mpm_water_dam_single_box` | 600 帧，1 次 | 共同有效格的底面平均压强累计归一化误差，以及立方体质心有效帧平均相对误差 |
| `fluid_grid_mpm_single_box` | 600 帧，1 次 | 相对初始 CIC 有效体积的逐帧有符号损失均值 |
| `fluid_grid_taylor_green` | 初态及 2000 帧，1 次 | 动能和耗散率相对 L2 误差，再对两者等权平均 |

扩展套件默认每组性能测三次，十万自由度场景每次计时 500 帧，溃坝和流固耦合计时 600 帧，TG 涡计时 2000 帧。性能采用同步墙钟计时。十万自由度场景关闭确定性，精度关闭 graph，性能开启 graph。

两个十万自由度场景精度运行三次，各端指标取均值后计算比值；其余场景精度运行一次。纯溃坝和 TG 涡关闭确定性；流固耦合开启确定性。性能关闭确定性，从初态启动三次独立计时试次；逐帧读取和落盘在精度试次内完成。流固耦合两种实现共用 Newton 参考，各版本分别汇总三次 FPS 和三次峰值显存，再对两个版本的汇总值取算术平均，与 Newton 比较。TG 涡可单独运行，FPS 比与显存比分母分别为 30 FPS 和 6144 MiB。

在 `cases/extended.json` 的纯溃坝条目中，可分别设置 `particle_error_scales.position_m` 和 `particle_error_scales.linear_velocity_mps`。省略时，两项分别按 m 和 m/s 取场景最短边对应的数值，当前为 1。报告保留本次完整精度轨迹和每次性能测量。粒子有显式 ID 时按 ID 对齐。两端均无 ID 时，仅在案例显式配置 implicit_particle_row_ids 后按行比较。扩展精度数据格式见[适配器协议](docs/adapter.md#扩展场景采集)。
