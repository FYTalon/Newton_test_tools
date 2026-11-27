# 评测适配器接口 v1

每家引擎提供一个适配器。适配器从请求里的 `case.id` 选择已有 example，按指定的求解器和运行设置构造一个场景实例。这个实例持有本次试验的模型、初态、求解器和控制器，由它的 `step()` 推进完整一帧。

参评双方公开实际用于测试的适配器源码、运行参数和标识映射。为每个引擎使用独立的 `adapters/<引擎名>/` 目录，并注明运行命令和引擎版本。Newton 参考端和 WanPhys 参评端都可以直接作为接入参考。[Newton 适配器](../adapters/newton/README.md) · [WanPhys 适配器](../adapters/wanphys/README.md)

## 接入仿真

SiMonitor 的示例先构造场景，再在步进循环中使用 `Monitor`。本工具的 `worker` 导入适配器，再把适配器提供的一次试验交给 `run_trial()`。通常可以复用同一引擎的读数函数，只为各场景指定入口、参数和必要的标识映射。[SiMonitor 接入说明](https://github.com/Nodevon-WS/SiMonitor#接入仿真) · [Worker 实现](../src/newton_test_tools/worker.py) · [测试流程](../src/newton_test_tools/monitor.py)

将 Python 模板复制为自己的适配器，替换场景工厂的导入和引擎名。在候选引擎的 Python 环境安装本工具，再将 `--adapter` 指向可导入的模块或 Python 文件。[Python 适配器模板](../adapters/example/adapter.py)

```sh
ENGINE_PY=/path/to/engine/.venv/bin/python
uv pip install --python "$ENGINE_PY" -e .
export CANDIDATE_ADAPTER="'$ENGINE_PY' -m newton_test_tools worker --adapter your_engine.benchmark_adapter"
```

适配器导出上下文管理器 `prepare_trial(request, *, output)`。先按 `case["id"]` 初始化已有场景，在构造求解器前传入两个开关，再 `yield Trial(...)`。上下文退出时释放场景资源。`output` 是本次试验的输出目录。由引擎自己的设备选择逻辑把 NVML 索引或 UUID 映射到仿真设备。模板中的 `create_scene(...)` 是引擎内部的场景工厂。

| `Trial` 字段 | 实现要求 |
| --- | --- |
| `engine`、`runtime` | 引擎显示名和下文所列的实际帧间隔、设备、场景规模及运行模式。 |
| `step()` | 推进完整一帧及其所有物理子步。 |
| `synchronize()` | 等待当次仿真提交在 GPU 上完成。 |
| `read_initial()` | 精度模式下返回 `model`、`shapes`、`initial_state` 与 `velocity_convention`。 |
| `observe()` | 精度模式下返回当前帧的 `(bodies, contact_pairs)`，使用 Newton 的参考标识。 |
| `warp_graphs()` | 可选。返回当前物理 Warp graph 的句柄，供工具核验预热时的实际重放。 |
| `check_final()` | 可选。写报告前核验引擎特有的状态或配置。 |

工具负责创建 `Monitor`、读取第 0 帧、逐帧步进与采集，以及性能预热、计时和写报告。Newton 和 WanPhys 使用同一个 `worker` 入口。提供了 `warp_graphs()` 时，工具在首个预热步记录实际重放次数，并在计时前恢复 Warp 原函数。

已有自定义采集循环时，可直接使用 `Monitor`。调用 `advance(dt_s=..., state=..., contact_pairs=...)` 提交同步后的数据，或用 `record_sim_time(t)` 按仿真时钟采样。已有同口径计时循环也可用 `record_performance(timed_frames, elapsed_seconds)` 提交原始计数。

公用运行器负责启动两端、重复试验、采样显存、比较结果和复核。其他语言可用自己的可执行入口，直接按下文的 JSON 协议生成文件。

## 进程协议

```sh
python -m newton_test_tools worker --adapter /path/adapter.py \
  --request /path/request.json --output /path/run
```

运行器开启一个新进程并为其创建输出目录。Python 默认使用上面的入口。其他语言可使用自己的可执行文件，接收相同的 `--request` 和 `--output`。试验完成后退出 0，在目录内写 `report.json` 及相应模式的文件。失败用非零退出码并输出原因。运行器收集 stdout 与 stderr 到 `worker.log`。一次调用运行一个场景的一次试验，性能模式下 GPU 仿真由这个进程执行。若入口是 shell 包装脚本，用 `exec` 使执行仿真的进程成为被监测的进程。

请求是 UTF-8 JSON。主要字段如下。

| 字段 | 内容 |
| --- | --- |
| `format` | `sim-bench-request-v1` |
| `mode` | `accuracy` 或 `performance` |
| `case` | 清单中的场景对象，包含 `id`、`solver`、`collision`、`world_count`、`frame_dt_seconds` 和 `substeps`。常规刚体提供 `normalization_floor` 和 `kinematic_scales`，扩展场景通过 `metric` 选择采样字段 |
| `protocol` | 本次运行使用的设置，字段见[六例清单](../cases/regular6.json)。常规精度用 `end_of_frame`，十万自由度用 `end_of_substep` |
| `gpu_devices` | 单元素数组，元素为 NVML 设备索引字符串或 GPU UUID |
| `reference_identity` | 常规刚体首次参考精度运行结束后提供的标识表，含 `format: sim-bench-identity-v1`、`bodies` 和 `shapes`。首次参考精度运行及性能运行的值为 `null` |
| `reference_snapshot` | MLS-MPM 请求提供的同轮 Newton 初态 `.npz` 路径 |

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
- `x_K` 对动态刚体逐帧计算质心位置、姿态、质心线速度和角速度误差。质心位置、质心线速度和角速度分别按清单中的 `position_m`、`linear_velocity_mps`、`angular_velocity_radps` 归一化。姿态误差为 `2 acos(|q_A · q_N|) / π`，工具用等价的半角公式稳定求值。取第 0 至 N 帧、所有动态体和四项误差的最大值。轨迹误差计算及归约采用 float32。
- 总误差 `x = max(x_C, x_D, x_K)`。清单的 `kinematic_scales` 对两端、全部动态体和采样帧统一。长度分别取球直径、传送带中心线半径、螺栓的 5 cm 参考尺度、摆杆长度、G1 初始骨盆高度和 ANYmal 初始机身高度。传送带、ANYmal 采用场景给定速度，其余速度尺度取 `sqrt(9.81 × L)`。角速度尺度取 `v_ref / L`。[常规六例清单](../cases/regular6.json)

## 性能数据

工具用 `protocol.warmup_frames` 预热，先同步设备再开始单次连续墙钟计时，之后每满 `sync_interval_frames` 帧同步一次。性能请求用 `duration_seconds` 指定目标时长，或用 `num_frames` 指定准确的计时帧数。固定帧数模式在最后一批同步后结束，尾批可少于同步间隔。预热沿用 example 的正常步进。计时段关闭渲染、逐帧接触和状态导出。性能报告的 `metrics.timed_frames` 和 `metrics.elapsed_seconds` 是原始计数，由运行器复算 FPS。RTF 定义为墙钟时间 / 仿真时间，作为补充信息。

运行器另写 `gpu-memory.json`，记录对应 GPU UUID、驱动、字节峰值、有效采样数、间隔和采样序列。使用 `Monitor.measure` 时，采样窗口由预热后的 GPU 同步与最后一帧后的 GPU 同步确定，计入正式运行期间仍驻留的初始化分配和缓存。缺少计时窗口或窗口内无采样点时，以 `scope: process` 保留全进程观测。正式清单开启 `require_timed_memory`，使用正式窗口的采样计算显存比，全进程观测只供诊断。NVML 查询间隔之间的瞬时分配可能没被捕获。若设备或场景规模核对失败，保留单端原始值，配对比值为空并给出原因。

性能重复次数由 `protocol.repetitions` 指定，适配器每次启动执行一次计时。常规六例每端默认 3 次，每次计时 500 帧。每次的报告和 NVML 峰值保存在 `run-N`，汇总 JSON 的 `trials` 保留各次原始值，`mean` 中的 FPS 为各轮总帧数除以总耗时，显存为各轮峰值的算术平均。FPS 比和显存比用候选端与参考端对应结果相除。旧清单省略重复次数时按一次处理。

每次场景步进算一帧，多 world 并行也按场景帧计数。相对 SiMonitor 的单端监控，这里的公用协议补充了配对运行、轨迹与碰撞形状对、参考标识映射，以及精度、FPS 和显存比。性能显存采用外部进程轮询。


## 扩展场景采集

扩展场景在 `Trial.diagnostic` 提供 `Diagnostic(model, initial, frame=...)` 或 `Diagnostic(model, initial, substeps=...)`。三个读数函数返回由 NumPy 数组或标量组成的字典。工具把模型写入 `model.npz`，把初态和每个样本顺次写入 `samples/000000.npz` 等文件。公用工具负责循环、核对步数、计算指标和保存报告。适配器用自己的引擎提供读数。

十万自由度的 `substeps` 是一个接收 `record(sample)` 的上下文管理器。进入时挂上求解器观察器，退出时卸载。每个物理子步恰好回调一次。刚体状态取子步推进后的动态刚体位姿和速度。接触距离、法向与力取自该子步所用的接触流形。法向压缩力统一为正。XPBD 用碰撞阶段的局部接触点恢复有符号距离，再读取求解后的接触力。MuJoCo 用实际送入求解的接触距离、接触帧和约束力重构法向力。在性能模式只调用完整帧 `step()`。

| `case.metric` | `model()` 和样本字段 |
| --- | --- |
| `conservation` | 模型提供动态刚体的 `mass`、`com_body`、`inertia_body`、`gravity`、`velocity_convention` 和 `effective_dof`。样本提供 `body_q`、`body_qd`、`contact_distance_m`、`normal_compression_N`，有接触时需同时提供同序的 `contact_world`、`shape0`、`shape1`、`contact_point_m`、`contact_normal` 和 `contact_force_N`。初态接触数组为空 |
| `particles` | 模型提供 FP32 `mass`，逐帧样本提供 FP32 `particle_q`、`particle_qd`，粒子顺序与封存参考一致 |
| `pressure_cube` | 模型提供粒子初态 SHA-256、总质量、诊断网格和压强方法标识及初始质心。逐帧样本含 `frame`、`time_end_s`、32×32 `floor_pressure_pa`（干格为 NaN）和 `cube_center_m`。使用 `coupled_summary` 的清单还需同网格的 `floor_contact_impulse_ns` 与 `floor_boundary_correction_impulse_ns`。Newton 端同时输出体积字段 |
| `volume` | 模型提供同样的粒子初态、质量与诊断网格元数据及体积方法标识。逐帧样本含 `frame`、`time_end_s`、`volume_ratio`、`initial_effective_volume_m3` 和 `nonfinite_count` |
| `tgv` | 默认在初态及各帧末提供标量 `energy`；若指定 `accuracy.sample_dt_star`，则在初态、选中的最近帧和终态提供 `energy` 及实际无量纲时刻 `t_star`。参考谱随运行副本保存 |

`pressure_cube` 和 `volume` 的参考请求使用同一个场景 ID。由 Newton 写出 `initial_snapshot.npz` 后，运行器把同轮快照的路径作为 `reference_snapshot` 传给 MLS-MPM。

压强由每格法向接触冲量及边界速度修正冲量之和除以采样格面积 0.01 m² 和帧间隔得到。对每帧两端共同有水的格点，逐格计算压强绝对差，累计全程后除以同一批格点的 Newton 压强绝对值之和。干湿掩码差异另外记录。立方体质心误差按每帧 Newton 质心距初始位置的位移归一化，再对有效帧取均值。版本二分别报告各端相对初始 CIC 有效体积的平均损失，报告保留逐帧有效体积和体积比。

十万自由度清单使用 `energy_scope: full_trajectory`。总线动量变化和机械能偏差均相对初值归一化，取全部物理子步的最大值。机械能取动能与重力势能之和。逐子步同时保留总质量、质心、质心角动量和关于全场质心的世界坐标惯性张量。穿透对全程满足 `d<0` 的接触点等权平均，再除以特征长度。负法向力取全程最大值并除以参考力。两端共享由 Newton 初态和清单固定参考长度、速度推得的尺度。比值排除 Newton 对应归一化值绝对值不超过 0.001 的项，其余与 1.05 比较。

机器狗群组在 `trials` 保留每轮物理检查。按引擎汇总各轮指标的算术平均到 `trial_means`，然后计算 `ratios`。逐子步曲线可查各轮的原始记录，顶层 `candidate` 与 `reference` 提供第一轮曲线。`verify` 按各次 `run.json` 保存的口径复算，包括旧版报告。

每端的 `events.jsonl` 按 world 和无序形状对记录连续的有效接触事件。起点取首次接触前的状态，终点取该对首次失去接触后的状态，记录全场动量、质心角动量和机械能的变化及事件内的穿透和负法向力峰值。持续到轨迹结束的事件仅记录起点及接触统计。

TG 涡按 `t*=0,0.01,…,19.99` 对齐已有谱参考。先在实际采样时刻对模拟动能做差分，内部点使用中心差分，端点使用单边差分；再将两条模拟曲线插值到参考时刻。参考耗散率取谱文件第三列。分别计算相对 L2 误差，再取算术平均。公开工具的 `verify` 从记录复算这些指标。
