#!/usr/bin/env bash
# Data-parallel serving on two Sparks: Basalt on both, spark_router in front (on this node).
#   SPARK_A=spark-a SPARK_B=spark-b API_KEY_FILE=~/.basalt-key scripts/dual/start-dual.sh
# Each Spark runs the whole model on its own 128 GB; the router keeps a conversation on the Spark that holds its
# prefix cache and spills to the other when one is busy. Backends listen on the direct link (192.168.100.x) or LAN.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
a=${SPARK_A:?ssh host of the first Spark}
b=${SPARK_B:?ssh host of the second Spark}
a_ip=${SPARK_A_IP:-192.168.100.1}
b_ip=${SPARK_B_IP:-192.168.100.2}
key=${API_KEY_FILE:?a file holding the API key the backends require}
# the repository on each Spark, relative to the ssh login directory (home) unless absolute
remote_root=${REMOTE_ROOT:-sparkstone}
for pair in "$a:$a_ip" "$b:$b_ip"; do
  host=${pair%%:*}; ip=${pair##*:}
  # $remote_root is left unquoted on the remote side so a leading ~ expands there
  # shellcheck disable=SC2029
  ssh "$host" "cd $remote_root && nohup env HOST=$ip API_KEY_FILE='$key' scripts/serve-spark.sh \
      > basalt-$host.log 2>&1 < /dev/null & echo started basalt on $host \($ip\)"
done
echo "waiting for both backends..."
for ip in "$a_ip" "$b_ip"; do
  until curl -fs "http://$ip:8080/health" >/dev/null; do sleep 2; done
  echo "  $ip healthy"
done
exec python3 "$root/router/spark_router.py" --backend "http://$a_ip:8080" --backend "http://$b_ip:8080" \
     --host "${ROUTER_HOST:-127.0.0.1}" --port "${ROUTER_PORT:-8000}" -v
