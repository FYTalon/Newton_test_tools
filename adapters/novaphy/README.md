# NovaPhy 六场景评测入口

候选端命令通过公用 `worker --adapter` 加载 `adapter.py`。评测器为每轮追加 `--request` 和 `--output`。[适配器协议](../../docs/adapter.md) · [六例清单](../../cases/regular6.json)

| 文件 | 用途 |
| --- | --- |
| `scenes.json` | 将六个 `case.id` 映射到 NovaPhy 的场景模块及其构造入口。 |
| `adapter.py` | 按请求创建试验，核验求解器和实际开关，并向 `worker` 提供 `Trial`。 |
| `snapshot.py` | 采集刚体初态、逐帧状态和有效接触对，使用 Newton 的参考标识。 |
| `robots.py`、`g1_shapes.json` | 初始化 G1、ANYmal，并提供 G1 的碰撞形状映射。 |

## 六例测试

从评测仓库根目录执行，将 `NOVAPHY_PY` 设为能运行 NovaPhy 的 Python 路径。

```sh
NOVAPHY_PY=/path/to/novaphy/.venv/bin/python
NEWTON_ADAPTER=".venv/bin/python -m newton_test_tools worker --adapter adapters/newton/adapter.py"
CANDIDATE_ADAPTER="'$NOVAPHY_PY' -m newton_test_tools worker --adapter '$PWD/adapters/novaphy/adapter.py'"

uv run --no-sync newton-test run --suite cases/regular6.json \
  --reference "$NEWTON_ADAPTER" --candidate "$CANDIDATE_ADAPTER" \
  --mode all --output runs/novaphy-regular6

uv run --no-sync newton-test verify runs/novaphy-regular6
```

六例的精度和性能每轮都按清单运行 500 帧。精度模式启用确定性、关闭 CUDA Graph，性能与显存模式关闭确定性、启用 CUDA Graph。性能报告记录预热后完成 500 帧的墙钟时间。
