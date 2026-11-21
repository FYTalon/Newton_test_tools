# 评测适配器接口 v1

每家引擎提供一个适配器入口。适配器从请求里的 `case.id` 选择已有 example，按指定的求解器和运行设置构造一个场景实例。这个实例持有本次试验的模型、初态、求解器和控制器，由它的 `step()` 推进完整一帧。

## 接入仿真

SiMonitor 的示例先构造场景，再在步进循环中使用 `Monitor`。这里也可直接用引擎已有的场景对象。性能接入需要步进和设备同步，精度还需要读取初态与帧末状态、接触对。通常可以复用同一引擎的读数函数，只为各场景指定入口、参数和必要的标识映射。 [SiMonitor 接入说明](https://github.com/Nodevon-WS/SiMonitor#接入仿真)

可以直接用公用模板作为 Python 入口。先在候选引擎的 Python 环境中安装本工具，再把 `--scene-module` 指向自己的场景模块。[独立适配器模板](../adapters/example/run.py)

```sh
ENGINE_PY=/path/to/engine/.venv/bin/python
uv pip install --python "$ENGINE_PY" .
export CANDIDATE_ADAPTER="$ENGINE_PY $PWD/adapters/example/run.py --scene-module your_engine.benchmark_scenes"
```

自己的模块实现 `create_scene(case, *, device, graph, deterministic, reference_identity)`。函数根据 `case["id"]` 选已有 example，在构造求解器前传入两个开关，并返回可关闭的场景上下文。用引擎自己的设备选择逻辑把 NVML 索引或 UUID 映射到仿真设备。模块需要能由候选环境导入。

| 场景接口 | 实现要求 |
| --- | --- |
| `frame_dt` | 实际完整帧时间，单位为秒。 |
| `step()` | 推进完整一帧及其所有物理子步。 |
| `synchronize()` | 等待当次仿真提交在 GPU 上完成。 |
| `runtime()` | 返回下文所列的实际设备、场景规模及运行模式。 |
| `read_initial()` | 精度模式下返回 `model`、`shapes`、`initial_state` 与 `velocity_convention`。 |
| `observe()` | 精度模式下返回当前帧的 `(bodies, contact_pairs)`，使用 Newton 的参考标识。 |

`adapters/example/run.py` 先创建场景，再把上述接口交给 `Monitor`。精度路径先提交第 0 帧，再执行 `with monitor.step(scene.frame_dt): scene.step()`。性能路径调用 `monitor.measure(scene.step)`，该路径只使用仿真步进和设备同步。

`Monitor` 在精度步进结束后调用同步与读数回调，在性能模式负责预热和计时。若已有采集循环，可调用 `advance(dt_s=..., state=..., contact_pairs=...)` 提交同步后的数据，或用 `record_sim_time(t)` 按仿真时钟采样。性能的 `first_warmup_step` 可用于核验 graph，已有同口径计时循环也可用 `record_performance(timed_frames, elapsed_seconds)` 提交原始计数。

公用运行器负责启动两端、重复试验、采样显存、比较结果和复核。其他语言可用自己的可执行入口，直接按下文的 JSON 协议生成文件。

## 运行入口

```sh
adapter --request /path/request.json --output /path/run
```

运行器开启一个新进程并为其创建输出目录。适配器完成后退出 0，并在目录内写 `report.json` 及相应模式的文件。失败用非零退出码并输出原因。运行器收集 stdout 与 stderr 到 `worker.log`。一次调用运行一个场景的一次试验，性能模式下 GPU 仿真由这个进程执行。若入口是 shell 包装脚本，用 `exec` 使适配器成为被监测的进程。

请求是 UTF-8 JSON。主要字段如下。

| 字段 | 内容 |
| --- | --- |
| `format` | `sim-bench-request-v1` |
| `mode` | `accuracy` 或 `performance` |
| `case` | 清单中的场景对象，包含 `id`、`solver`、`collision`、`world_count`、`frame_dt_seconds`、`substeps` 及 `normalization_floor` |
| `protocol` | 本次运行使用的设置，字段见[六例清单](../cases/regular6.json)。精度采样阶段固定为 `end_of_frame` |
| `gpu_devices` | 单元素数组，元素为 NVML 设备索引字符串或 GPU UUID |
| `reference_identity` | 首次参考精度运行结束后提供的标识表，含 `format: sim-bench-identity-v1`、`bodies` 和 `shapes`。首次参考精度运行及性能运行的值为 `null` |

适配器在创建场景和求解器前按 `protocol.deterministic` 与 `protocol.graph` 设置对应的引擎选项，并在创建完成后核验实际配置。对于 Newton，设置包括求解器模式、外部碰撞排序和 graph 构建。其他引擎按自身接口实现对应设置。

每端 `report.json` 的 `format` 为 `sim-bench-result-v1`，同时提供 `engine`（显示名称）、`case`（场景 ID）、`mode`、`runtime`、`metrics`。运行器核对实际 `runtime` 的 `device_uuid`、`frame_dt_seconds`、`substeps`、`world_count`、`solver`、`collision`、`deterministic` 和 `graph`。有 graph 模式的引擎按请求设置，未提供这种执行模式的引擎将 `runtime.graph` 写为 `null`。精度还要求 `contact_stage`。性能还要求 `model_counts`，记录场景中实际比较的 `world_count`、`body_count`、`joint_count`、`joint_dof_count`、`particle_count` 和 `collidable_shape_count`。可附加引擎版本、example 路径及哈希等来源信息。

## 精度数据

`trajectory.json` 的 `format` 为 `sim-bench-rigid-trace-v1`。所有物理量采用 SI 单位。示例中的 `box` 和 `ground` 是标准名称。实际以本轮参考引擎导出的 `identity.json` 为准。

| 字段 | 内容 |
| --- | --- |
| `case`、`solver`、`frame_dt_seconds`、`num_frames` | 对应本次运行 |
| `velocity_convention` | `com_world` 表示线速度以质心为参考，`origin_world` 表示线速度以刚体原点为参考。线速度和角速度均用世界坐标表达 |
| `model` | 一次性刚体表。每行包含整数 `world`、字符串 `name`、`mass`、`inv_mass`、本体坐标的 `com_body` 三向量、关于质心的 `inertia_body` 3×3 张量、`flags`（0 静态、1 动态、2 运动学） |
| `shapes` | 一次性可碰撞形状表。每行含 `world`、`name`、所属刚体标准名 `body`（地面等无刚体几何用 `null`）与 `type` |
| `states` | 按帧顺序排列的第 0 至 N 帧。每帧含 `frame` 和 `bodies`。每个刚体含 `world`、`name`、`q`（刚体原点位置 xyz 和本体到世界的四元数 xyzw）、`qd`（线速度 xyz 和角速度 xyz） |
| `pairs` | 第 1 至 N 帧的对象，键为十进制帧号字符串。值为该帧实际生成的有效形状对列表，例如 `[[0, "box", "ground"]]`。双方都无接触时各自应写 `[]` |

`world` 从 0 开始。两端的刚体与形状用 `(world, name)` 标识。共享静态刚体可用 world `-1`，对应的共享碰撞形状在每个可发生接触的 world 中各列一次。适配器将本地索引转换成参考标识。形状分类 v1 采用 0 未分类、1 平面、2 高度场、3 球、4 胶囊、5 椭球、6 圆柱、7 盒、8 网格、9 圆锥、10 凸网格、11 高斯几何。

每次完整 `step()` 结束并同步 GPU，再读取该帧状态和求解流程中的接触缓冲。统一用帧末保留的有效碰撞记录。宽相候选、无效槽位和跨 world 对不进入列表。采集过程中不额外重跑碰撞，也不跨子步求并集。评分器按无序对排序去重。无法提供某一帧接触数据时返回错误，空集合只表示这一帧无有效接触。用已封存的参考形状表核对映射，不依赖两端数组存放顺序。

`metrics` 在精度模式提供 `completed_frames` 和 `simulation_seconds`。比较器还核对初态、刚体质量、逆质量、质心与惯性。场景资产、几何尺寸、关节、材质、驱动和策略在封存场景时独立核对。

### 误差计算

误差报告采用 `sim-bench-accuracy-v2`，读取本协议的 v1 轨迹。

- `x_C` 为逐帧形状对集合的 Jaccard 误差最大值。分子为对称差数量，分母为并集数量。并集为空时该帧为 0。
- `x_D` 用相邻帧质心线动量差及世界坐标下关于质心的角动量差除以帧间隔，分别得等效合力和等效合力矩。将 `xyzw` 单位四元数转为本体到世界的旋转矩阵 R，世界惯性为 R I Rᵀ，其中矩阵第 3 行第 1 列为 2(xz − yw)。每个动态刚体各项尺度取参考轨迹相应范数峰值与场景正下限的较大值。用同一参考尺度算两端差的归一化范数，取所有帧、刚体和两类量的最大值。
- `x_K` 逐帧计算动态刚体的质心位置、质心线速度、角速度之差。固定尺度分别取参考端相对初始位置的最大位移、最大速度范数及最大角速度范数与各自正下限的较大值。取全轨迹三类归一化差的最大值。计算包含第 0 帧。轨迹误差计算及归约采用 float32。

## 性能数据

适配器用 `protocol.warmup_frames` 预热，先同步设备再开始单次连续墙钟计时，之后每满 `sync_interval_frames` 帧同步一次，达到 `duration_seconds` 时结束。预热沿用 example 的正常步进。计时段关闭渲染、逐帧接触和状态导出。性能报告的 `metrics.timed_frames` 和 `metrics.elapsed_seconds` 是原始计数，由运行器复算 FPS。RTF 定义为墙钟时间 / 仿真时间，作为补充信息。

运行器另写 `gpu-memory.json`，记录对应 GPU UUID、驱动、字节峰值、有效采样数和间隔。显存峰值包含建场景、graph 创建、预热和正式运行期间观测到的进程驻留量。NVML 查询间隔之间的瞬时分配可能没被捕获。若设备或场景规模核对失败，保留单端原始值，配对比值为空并给出原因。

性能重复次数由 `protocol.repetitions` 指定，适配器每次启动仍只执行一次计时。常规六例每端为 3 次。每次的报告和 NVML 峰值保存在 `run-N`，汇总 JSON 的 `trials` 保留各次原始值，`mean` 给出 FPS 和各次显存峰值的算术平均。FPS 比和显存比分别用候选端与参考端的对应均值相除。旧清单省略此字段时按一次处理。

每次场景步进算一帧，多 world 并行也按场景帧计数。相对 SiMonitor 的单端监控，这里的公用协议补充了配对运行、轨迹与碰撞形状对、参考标识映射，以及精度、FPS 和显存比。性能显存采用外部进程轮询。
