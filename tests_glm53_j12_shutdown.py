"""J12 host test: ZMQOffloadServer open/close 30 times with its thread blocked in poll (and once mid-request): every
close must return, the thread must exit and close its own socket (no libzmq assertion / abort)."""
import os, sys, tempfile, threading, time, types
sys.path.insert(0, os.environ.get("LMC_TREE", "/opt/glm53/LMCache-tt-j12"))
import lmcache, zmq, msgspec
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.offload_server.message import OffloadMsg
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache, get_zmq_context, get_zmq_socket
assert lmcache.__file__.startswith(os.environ.get("LMC_TREE", "/opt/glm53/LMCache-tt-j12")), lmcache.__file__
os.environ["LMCACHE_OFFLOAD_RPC_PORT"] = "177"
stored = []
eng = types.SimpleNamespace(metadata=types.SimpleNamespace(engine_id="j12test"), store=lambda **kw: stored.append(kw))
t0 = time.time()
for i in range(30):
    s = ZMQOffloadServer(eng, tp_rank=0)
    if i % 10 == 9:  # one real request round trip before the close
        path = get_zmq_rpc_path_lmcache("j12test", "offload", 177, 0)
        c = get_zmq_socket(get_zmq_context(use_asyncio=False), path, "ipc", zmq.REQ, "connect")
        c.send(msgspec.msgpack.encode(OffloadMsg(hashes=[1], slot_mapping=[0], offsets=[0])))
        assert c.poll(5000), "no reply"
        c.recv(); c.close(linger=0)
    time.sleep(0.05 * (i % 3))
    s.close()
    assert not s.thread.is_alive(), f"cycle {i}: thread alive after close"
    assert s.socket.closed, f"cycle {i}: socket left open"
print(f"J12_SHUTDOWN_TEST PASS: 30 open/close cycles, {len(stored)} requests served, {time.time() - t0:.1f} s")
