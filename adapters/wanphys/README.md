# WanPhys 适配器

`run.py` 是公开评测的 WanPhys 参评端入口。它用 `scenes.json` 选择六例的模块和参数，将请求里的确定性和 graph 布尔值传给例子的 `create_scene(argv)`，并检查实际求解器、碰撞配置与 graph 状态。[场景配置及标识映射](scenes.json)

## 运行

准备可独立运行的 WanPhys Python 环境。六个例子需提供 `create_scene(argv)`，并接受 `--deterministic`、`--no-deterministic`、`--graph`、`--no-graph`。场景和资产由该环境提供。

从公用工具仓库根目录执行，将 `WANPHYS_PY` 设为对应环境的 Python。

```sh
WANPHYS_PY=/path/to/WanPhys/.venv/bin/python
uv pip install --python "$WANPHYS_PY" .
NEWTON_ADAPTER=".venv/bin/python adapters/newton/run.py"
CANDIDATE_ADAPTER="'$WANPHYS_PY' '$PWD/adapters/wanphys/run.py'"
```

将这两个命令传给主 README 的短测或正式评测。两端可以使用独立 Python 环境。WanPhys 适配器在进程内使用 `cuda:0`。[接入测试命令](../../README.md#4-接入自己的引擎测试)

## 采集逻辑

- 精度在第 0 帧和每个完整帧结束后读取刚体状态，形状对取帧末保留的有效碰撞记录。公开的名称映射将本地编号对齐到 Newton。MuJoCo 在独立状态中通过 FK 重建世界质心速度，采集过程只读求解状态。[读取源码](snapshot.py)
- 性能首个预热步检查物理 graph 是否实际重放。计时调用场景的 `step()` 和设备同步，显存由运行器通过 NVML 采样。报告记录实际参数、solver 模式、example 和场景配置哈希。[运行源码](run.py)
