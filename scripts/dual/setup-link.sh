#!/usr/bin/env bash
# Configure the direct ConnectX-7 cable between two DGX Sparks (run on each node, as root).
#
# GB10 wires the ConnectX-7 over two PCIe Gen5 x4 links, so each physical QSFP port shows up as TWO interfaces
# (e.g. enp1s0f1np1 and enP2p1s0f1np1), each good for ~100 Gb/s. Both need an address for the full 200 Gb/s.
# Subnets and MTU follow NVIDIA's connect-two-sparks playbook (192.168.100.0/24 + 192.168.101.0/24, MTU 9000
# for a direct two-node link).
#
#   sudo NODE=1 scripts/dual/setup-link.sh          # first Spark  -> .1 on both subnets
#   sudo NODE=2 scripts/dual/setup-link.sh          # second Spark -> .2 on both subnets
#   PERSIST=1 also writes /etc/netplan/60-sparkstone-link.yaml (otherwise the addresses last until reboot)
#   IF_A / IF_B override the interface names (see `ip -br link` and `ibdev2netdev`)
set -euo pipefail
node=${NODE:?set NODE=1 or NODE=2}
if_a=${IF_A:-enp1s0f1np1}
if_b=${IF_B:-enP2p1s0f1np1}
mtu=${MTU:-9000}
for i in "$if_a" "$if_b"; do
  ip link show "$i" >/dev/null 2>&1 || { echo "no interface $i; set IF_A/IF_B (ip -br link)" >&2; exit 1; }
done
ip link set "$if_a" mtu "$mtu" up
ip link set "$if_b" mtu "$mtu" up
ip addr replace "192.168.100.$node/24" dev "$if_a"
ip addr replace "192.168.101.$node/24" dev "$if_b"
if [ "${PERSIST:-0}" = 1 ]; then
  cat > /etc/netplan/60-sparkstone-link.yaml <<YAML
network:
  version: 2
  ethernets:
    $if_a:
      mtu: $mtu
      addresses: [192.168.100.$node/24]
    $if_b:
      mtu: $mtu
      addresses: [192.168.101.$node/24]
YAML
  chmod 600 /etc/netplan/60-sparkstone-link.yaml
  netplan apply
fi
peer=$(( node == 1 ? 2 : 1 ))
echo "node $node: $if_a=192.168.100.$node $if_b=192.168.101.$node (MTU $mtu); peer is .$peer"
ping -c 2 -W 1 "192.168.100.$peer" >/dev/null && echo "peer reachable on 192.168.100.$peer" || echo "peer not answering yet"
