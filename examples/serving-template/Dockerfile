# 多阶段构建：独立部署用。复制本目录为独立仓库后按需改 base 镜像。
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum* ./
# monorepo 开发时需一并复制 leaves 源码；独立仓则只有本模块
COPY . .
# 独立仓：去掉 replace 后 go mod download
RUN go mod download 2>/dev/null || true
RUN CGO_ENABLED=0 go build -o /out/leaves-serving .

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /
COPY --from=build /out/leaves-serving /leaves-serving
# 运行时挂载模型到 /models
ENV LEAVES_HTTP_ADDR=:8080
ENV LEAVES_MODEL=/models/model.leaves.json
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/leaves-serving"]
