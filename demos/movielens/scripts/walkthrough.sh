#!/usr/bin/env bash
# MovieLens Ranker Agent 端到端 walkthrough
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
echo "== leaves MovieLens walkthrough =="
echo "root: $ROOT"

agent() {
  echo ""
  echo ">> agent $*"
  go run ./demos/movielens/cmd/agent "$@"
}

agent status
agent prepare
agent train -objective rank:ndcg -rounds 20 -depth 4
agent eval
agent recommend -group 0 -topk 10

echo ""
echo "== done =="
echo "see demos/movielens/out/ and TUTORIAL.md"
