#!/usr/bin/env python3
"""Report network topology and status for TorchSpec multi-node setups.

Checks:
  1. Available RDMA devices and their link rate/state/layer on each node.
  2. Network interfaces suitable for NCCL_SOCKET_IFNAME on each node.
  3. Pairwise TCP connectivity between all nodes in the Ray cluster.

Usage (local node only):
    python tools/check_network_topology.py

Usage (full Ray cluster):
    RAY_ADDRESS=<head-node-ip>:<port> python tools/check_network_topology.py
"""

import os
import socket
import subprocess
import time


def _read_sysfs(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def get_rdma_devices() -> list[dict]:
    ib_root = "/sys/class/infiniband"
    devices = []

    if not os.path.isdir(ib_root):
        result = subprocess.run(["ibv_devinfo"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return devices
        current: dict | None = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("hca_id:"):
                if current:
                    devices.append(current)
                current = {
                    "name": line.split()[-1],
                    "port": 1,
                    "state": "unknown",
                    "rate": "unknown",
                    "link_layer": "unknown",
                    "phys_state": "unknown",
                }
            elif current and line.startswith("transport:"):
                current["link_layer"] = line.split()[-1]
            elif current and line.startswith("state:"):
                current["state"] = line.split(":", 1)[-1].strip()
            elif current and line.startswith("active_width"):
                pass
        if current:
            devices.append(current)
        return devices

    for dev_name in sorted(os.listdir(ib_root)):
        ports_path = os.path.join(ib_root, dev_name, "ports")
        if not os.path.isdir(ports_path):
            continue
        for port_num in sorted(os.listdir(ports_path)):
            port_path = os.path.join(ports_path, port_num)
            state = _read_sysfs(os.path.join(port_path, "state"))
            rate = _read_sysfs(os.path.join(port_path, "rate"))
            link_layer = _read_sysfs(os.path.join(port_path, "link_layer"))
            phys_state = _read_sysfs(os.path.join(port_path, "phys_state"))
            devices.append(
                {
                    "name": dev_name,
                    "port": int(port_num),
                    "state": state,
                    "rate": rate,
                    "link_layer": link_layer,
                    "phys_state": phys_state,
                }
            )

    return devices


def _rdma_backed_interfaces() -> set[str]:
    ib_root = "/sys/class/infiniband"
    ifaces: set[str] = set()
    if not os.path.isdir(ib_root):
        return ifaces
    for dev_name in os.listdir(ib_root):
        net_path = os.path.join(ib_root, dev_name, "device", "net")
        if os.path.isdir(net_path):
            for iface in os.listdir(net_path):
                ifaces.add(iface)
    return ifaces


def _iface_ipv4(iface: str) -> str:
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", iface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                return line.split()[1].split("/")[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def get_nccl_interfaces() -> list[dict]:
    net_root = "/sys/class/net"
    if not os.path.isdir(net_root):
        return []

    rdma_ifaces = _rdma_backed_interfaces()
    skip_prefixes = ("lo", "docker", "veth", "virbr", "br-")
    results = []

    for iface in sorted(os.listdir(net_root)):
        if any(iface.startswith(p) for p in skip_prefixes):
            continue
        operstate = _read_sysfs(os.path.join(net_root, iface, "operstate"))
        if operstate != "up":
            continue
        ip_addr = _iface_ipv4(iface)
        results.append(
            {
                "name": iface,
                "operstate": operstate,
                "ip": ip_addr,
                "rdma_backed": iface in rdma_ifaces,
            }
        )

    return results


def print_rdma_report(devices: list[dict], node_label: str = "local") -> None:
    print(f"\nRDMA Devices on {node_label}:")
    if not devices:
        print("  No RDMA devices found.")
        return
    col = "{:<16} {:>5}  {:<22} {:<28} {:<16} {:<20}"
    print("  " + col.format("Device", "Port", "State", "Rate", "Link Layer", "Phys State"))
    print("  " + "-" * 110)
    for d in devices:
        print(
            "  "
            + col.format(
                d["name"], d["port"], d["state"], d["rate"], d["link_layer"], d["phys_state"]
            )
        )


def print_nccl_report(ifaces: list[dict], node_label: str = "local") -> None:
    print(f"\nNetwork Interfaces for NCCL_SOCKET_IFNAME on {node_label}:")
    if not ifaces:
        print("  No suitable UP interfaces found.")
        return
    col = "{:<20} {:<16} {:<10}"
    print("  " + col.format("Interface", "IP", "RDMA-backed"))
    print("  " + "-" * 50)
    for i in ifaces:
        print("  " + col.format(i["name"], i["ip"] or "n/a", "yes" if i["rdma_backed"] else "no"))

    rdma_names = [i["name"] for i in ifaces if i["rdma_backed"]]
    candidates = rdma_names if rdma_names else [i["name"] for i in ifaces]
    if candidates:
        print(f"\n  Recommended: export NCCL_SOCKET_IFNAME={candidates[0]}")


def _local_probe_info() -> dict:
    return {
        "rdma_devices": get_rdma_devices(),
        "nccl_interfaces": get_nccl_interfaces(),
    }


PROBE_PORT = 29500
CONNECT_TIMEOUT = 5.0


def _tcp_server_listen(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(64)
    s.settimeout(CONNECT_TIMEOUT * 2)
    return s


def _tcp_probe(target_ip: str, port: int) -> tuple[bool, float]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TIMEOUT)
    t0 = time.monotonic()
    try:
        s.connect((target_ip, port))
        rtt = (time.monotonic() - t0) * 1000
        s.close()
        return True, rtt
    except OSError:
        return False, -1.0


def run_local() -> None:
    rdma = get_rdma_devices()
    ifaces = get_nccl_interfaces()
    print_rdma_report(rdma)
    print_nccl_report(ifaces)


def run_cluster() -> None:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    ray.init(address=ray_address, ignore_reinit_error=True)

    live_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    if not live_nodes:
        print("No live Ray nodes found.")
        return

    print(f"Ray cluster: {len(live_nodes)} live node(s)")

    @ray.remote(num_cpus=0)
    class NetworkProbeActor:
        def gather_info(self) -> dict:
            return _local_probe_info()

        def open_server(self, port: int) -> str:
            self._server = _tcp_server_listen(port)
            return socket.gethostbyname(socket.gethostname())

        def accept_all(self, count: int) -> None:
            for _ in range(count):
                try:
                    conn, _ = self._server.accept()
                    conn.close()
                except OSError:
                    pass
            self._server.close()

        def probe(self, target_ip: str, port: int) -> tuple[bool, float]:
            return _tcp_probe(target_ip, port)

    actors = []
    for node in live_nodes:
        node_id = node["NodeID"]
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        actor = NetworkProbeActor.options(scheduling_strategy=strategy).remote()
        actors.append((node["NodeManagerAddress"], actor))

    infos = ray.get([a.gather_info.remote() for _, a in actors])

    for (node_ip, _), info in zip(actors, infos):
        print_rdma_report(info["rdma_devices"], node_label=node_ip)
        print_nccl_report(info["nccl_interfaces"], node_label=node_ip)

    print("\nPairwise TCP Connectivity Matrix:")
    n = len(actors)
    node_ips_list = [ip for ip, _ in actors]

    server_ips = ray.get([a.open_server.remote(PROBE_PORT) for _, a in actors])

    accept_futures = [a.accept_all.remote(n - 1) for _, a in actors]

    probe_futures = {}
    for i, (_, src_actor) in enumerate(actors):
        for j, tgt_ip in enumerate(server_ips):
            if i == j:
                continue
            fut = src_actor.probe.remote(tgt_ip, PROBE_PORT)
            probe_futures[(i, j)] = fut

    probe_results = {k: ray.get(v) for k, v in probe_futures.items()}
    ray.get(accept_futures)

    header = "{:<18}".format("src \\ dst")
    for ip in node_ips_list:
        header += "{:>22}".format(ip)
    print(header)
    print("-" * (18 + 22 * n))
    for i in range(n):
        row = "{:<18}".format(node_ips_list[i])
        for j in range(n):
            if i == j:
                row += "{:>22}".format("self")
            else:
                ok, rtt = probe_results[(i, j)]
                cell = f"OK {rtt:.1f}ms" if ok else "FAIL"
                row += "{:>22}".format(cell)
        print(row)


if __name__ == "__main__":
    ray_address = os.environ.get("RAY_ADDRESS", "")
    if ray_address:
        run_cluster()
    else:
        run_local()
