# Newton_test_tools

## 1. 工具说明

`newton-test` 通过命令行适配器运行参考引擎和参评引擎。适配器用各自已有的 example 构建场景、推进仿真并导出数据。公用工具负责启动两端、计算误差、汇总 FPS 和峰值显存，以及复核保存的结果。两端可以使用各自的运行环境和编程语言。

当前评测覆盖以下六个刚体与铰链体场景。求解器、帧时间、子步数和归一化尺度见[六例清单](cases/regular6.json)。

| 场景 ID | 求解器 | world 数 | 每帧子步数 |
| --- | --- | ---: | ---: |
| `basic_shapes` | XPBD | 1 | 5 |
| `basic_conveyor` | XPBD | 1 | 10 |
| `nut_bolt_hydro` | XPBD | 20 | 4 |
| `basic_pendulum` | XPBD | 1 | 10 |
| `robot_g1` | MuJoCo 外部碰撞 | 4 | 6 |
| `robot_anymal_c_walk` | MuJoCo 外部碰撞 | 1 | 4 |

| 模式 | 确定性 | 场景级 graph | 每例每端默认运行 |
| --- | --- | --- | --- |
| 精度 | 开 | 关 | 从相同初态连续运行 500 帧，独立运行 2 次 |
| 性能 | 关 | 有此模式时开启 | 预热 5 帧后计时 600 秒，独立运行 3 次 |

精度用第一次轨迹计算碰撞形状对误差 `x_C`、动力学响应误差 `x_D` 和运动学误差 `x_K`，第二次用来检查该端是否可重复。性能在同一张 GPU 上顺序运行，每 5 帧同步一次。运行器用 NVML 每 10 ms 采样进程显存，包含初始化和预热阶段。FPS 比和显存比分别用参评端和参考端的三次均值相除。

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
NEWTON_ADAPTER=".venv/bin/python adapters/newton/run.py"

uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$NEWTON_ADAPTER" \
  --frames 8 --repetitions 1 --duration 2 --performance-repetitions 1 \
  --output runs/newton-smoke

uv run --no-sync newton-test report runs/newton-smoke
uv run --no-sync newton-test verify runs/newton-smoke
```

自检会生成两端轨迹、计时和显存记录。`verify` 会检查文件哈希并重算报告。Newton 版本、资产来源和 Hydro 补丁说明见[参考适配器文档](adapters/newton/README.md)。

## 3. 写适配器（adapter）

参评双方都应公开实际用于测试的适配器源码、场景运行参数和标识映射，便于按同一套实现复验。每个引擎的适配器放在 `adapters/<引擎名>/`。本库已提供 Newton 参考端和 WanPhys 参评端的完整适配器。[Newton 适配器](adapters/newton/run.py) · [WanPhys 适配器](adapters/wanphys/run.py) · [WanPhys 场景配置](adapters/wanphys/scenes.json)

每次试验由公用工具启动一个适配器进程，并追加 `--request` 和 `--output`。

```sh
your-adapter --request /path/to/request.json --output /path/to/run
```

用请求里的 `case.id` 选择已有 example，按 `case` 读取求解器和物理配置。在构造场景前应用 `protocol.deterministic` 和 `protocol.graph`，并按 `gpu_devices` 选择设备。精度模式下，用 `reference_identity` 将本地刚体和形状编号映射到 Newton 的 `(world, name)` 标识。

Python 可以直接复用仓库里的入口，配套提供一个能导入的模块，如 `your_engine.benchmark_scenes`。模块实现 `create_scene(case, *, device, graph, deterministic, reference_identity)`，返回一个在进入时构造场景、退出时释放资源的上下文管理器。场景提供以下接口。[Python 适配器模板](adapters/example/run.py)

| 场景接口 | 提供的能力 |
| --- | --- |
| `frame_dt`、`step()` | 报告帧间隔，推进完整一帧及全部物理子步。 |
| `synchronize()` | 等待 GPU 完成仿真。 |
| `runtime()` | 报告实际设备 UUID、求解器、场景规模和两个开关。 |
| `read_initial()` | 精度模式下提供模型、形状表、第 0 帧状态和速度约定。 |
| `observe()` | 精度模式下返回帧末的 `(bodies, contact_pairs)`。 |

精度先记录第 0 帧，再逐帧读取刚体状态和求解过程中留下的有效形状对。性能计时只执行步进和设备同步。Python 的 `Monitor` 会生成协议文件，其他语言可以直接写 `report.json` 和精度模式所需的 `trajectory.json`。模型、接触、计时及输出字段见[适配器协议](docs/adapter.md)。

## 4. 接入自己的引擎测试

在本工具根目录把候选命令指向自己的引擎。使用 Python 模板时，将下面的 Python 路径和模块名换成自己的，模块须能从该环境导入。

```sh
ENGINE_PY=/path/to/engine/.venv/bin/python
uv pip install --python "$ENGINE_PY" .
CANDIDATE_ADAPTER="'$ENGINE_PY' '$PWD/adapters/example/run.py' --scene-module your_engine.benchmark_scenes"
NEWTON_ADAPTER=".venv/bin/python adapters/newton/run.py"
```

已有独立入口时，可将 `CANDIDATE_ADAPTER` 设为其完整命令，例如 `/path/to/engine/bin/benchmark_adapter`。运行器会追加本次试验的请求和输出参数。先用 `basic_shapes` 跑 100 帧短测，检查首次接触后的形状对映射，再跑正式六例。

以 WanPhys 作为参评端时，将 `CANDIDATE_ADAPTER` 指向公开的 `adapters/wanphys/run.py`，用已安装 WanPhys 的 Python 环境运行。[WanPhys 运行说明](adapters/wanphys/README.md)

```sh
uv run --no-sync newton-test run --suite cases/regular6.json --case basic_shapes \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --frames 100 --repetitions 1 --duration 2 --performance-repetitions 1 \
  --output runs/candidate-smoke

uv run --no-sync newton-test verify runs/candidate-smoke
```

首次运行含 `nut_bolt_hydro` 的正式精度前，对上面检出的 Newton 应用随附补丁。正式命令省略短测的覆盖参数，使用清单规定的帧数、时长和重复次数。[Hydro 补丁说明](adapters/newton/README.md#20-world-hydro-的确定性参考)

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

输出目录的 `report.md` 汇总结果，`report.json` 保留逐帧误差和性能明细，`run.json` 记录实际测试配置。两端的请求、轨迹、显存采样和日志可在 `cases/<场景>/<模式>/` 下查到。

可以用已有轨迹离线复算精度：

```sh
uv run --no-sync newton-test compare --suite cases/regular6.json --case basic_shapes --frames 8 \
  --reference runs/newton-smoke/cases/basic_shapes/accuracy/reference/run-1/trajectory.json \
  --candidate runs/newton-smoke/cases/basic_shapes/accuracy/candidate/run-1/trajectory.json
```
