# leaves v2.5.3

**主题**：`leaves version` 子命令——装了哪个版本，一眼可查。

## Highlights

1. **`leaves version`**：输出 `{version, go[, commit]}` JSON（`debug.ReadBuildInfo`）。`go install ...@vX.Y.Z` 安装的二进制带真实 tag；仓库内 `go build/run` 显示 `(devel)`。来源：v2.5.2 发版验证时 `leaves version` 报「未知子命令」——用户/Agent 排查「装的是不是最新」无入口。

## Usage

```powershell
> leaves version
{"go":"go1.26.5","version":"v2.5.3"}
```

## Verification

- `TestVersionDoc`：version/go 字段必存在。
- 仓库内 `go run ./cmd/leaves version` 实测输出 JSON。
- `go test ./cmd/leaves ./docs -count=1` 绿。

## Compatibility

- **Breaking**：无。新增子命令；usage 的合法子命令提示已同步。
