module github.com/example/leaves-serving

go 1.26.0

// 本模板可整目录复制为独立仓库（独立产品仓已就绪：
// github.com/linkerlin/leaves-serving）。复制后：
//  1. 改 module 路径
//  2. 删除下方 replace，改为 require 具体 leaves 版本，例如：
//     require github.com/linkerlin/leaves/v2 v2.6.2
//  3. go mod tidy

require github.com/linkerlin/leaves/v2 v2.6.2

require (
	github.com/born-ml/born v0.9.1 // indirect
	github.com/go-webgpu/goffi v0.5.2 // indirect
	github.com/go-webgpu/webgpu v0.5.1 // indirect
	github.com/gogpu/gputypes v0.5.0 // indirect
	github.com/gogpu/naga v0.17.13 // indirect
	github.com/gogpu/wgpu v0.29.0 // indirect
	github.com/toitware/ubjson v0.0.0-20260115144145-354787fca6c1 // indirect
	golang.org/x/sys v0.45.0 // indirect
)

replace github.com/linkerlin/leaves/v2 => ../..
