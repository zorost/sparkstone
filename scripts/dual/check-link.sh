#!/usr/bin/env bash
# Measure the Spark-to-Spark link: what the dual-Spark modes in docs/03-dual-spark.md depend on.
#   Peer side first:  scripts/dual/check-link.sh server
#   This side:        PEER=192.168.100.2 scripts/dual/check-link.sh client
# Reports: ping RTT at MTU 9000, RDMA write bandwidth on both halves, and RDMA write LATENCY for small messages
# (8 B - 64 KiB). The latency is the go/no-go number for expert parallelism: see the table in docs/03.
# Needs perftest (ib_write_bw / ib_write_lat): sudo apt install perftest
set -euo pipefail
mode=${1:?server or client}
hca_a=${HCA_A:-rocep1s0f1}
hca_b=${HCA_B:-roceP2p1s0f1}
out=${OUT:-link-report-$(date +%Y%m%d-%H%M%S).txt}
if [ "$mode" = server ]; then
  echo "serving perftest on $hca_a (ports 18515/18516) and $hca_b (18517/18518); Ctrl-C to stop"
  while true; do
    ib_write_bw -d "$hca_a" -p 18515 --report_gbits -q 4 >/dev/null 2>&1 &
    ib_write_bw -d "$hca_b" -p 18517 --report_gbits -q 4 >/dev/null 2>&1 &
    wait
    ib_write_lat -d "$hca_a" -p 18516 -a >/dev/null 2>&1
  done
fi
peer=${PEER:?set PEER to the other Spark's 192.168.100.x address}
peer_b=${PEER_B:-${peer/192.168.100./192.168.101.}}
{
  echo "# sparkstone link report $(date -Is) $(hostname) -> $peer / $peer_b"
  echo "## ping (MTU 9000 payload)"
  ping -c 20 -i 0.05 -M do -s 8972 "$peer" | tail -2
  echo "## RDMA write bandwidth, both halves at once (Gb/s)"
  ib_write_bw -d "$hca_a" -p 18515 --report_gbits -q 4 "$peer" | tail -3 &
  ib_write_bw -d "$hca_b" -p 18517 --report_gbits -q 4 "$peer_b" | tail -3 &
  wait
  sleep 1
  echo "## RDMA write latency by message size (usec): the expert-parallel gate"
  ib_write_lat -d "$hca_a" -p 18516 -a "$peer" | grep -E '^ *[0-9]' | awk '$1<=65536'
} | tee "$out"
echo "saved $out; add it to bench/results/ with your benchmark rows"
cat <<'NOTE'
Reading the latency column (t_typical, 8 B - 8 KiB):
  <= 10 us   expert parallel can beat one Spark per stream (docs/03: ~1.6x on drafted windows)
  10-50 us   marginal; data parallel stays the default
  > 100 us   check C-states / power profile (forum reports of 1-1.5 ms at default settings), then re-measure
NOTE
