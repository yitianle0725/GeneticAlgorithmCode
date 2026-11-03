"""结果写入互斥：进程退出后由操作系统释放，避免空目录锁永久阻止续跑。"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


PROTOCOL = "os_file_lock_v1"


@contextmanager
def _writer_lock(path: Path):
    # 此文件必须保留；删除后新进程可能锁住另一个文件，破坏互斥。
    with path.open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"结果正在写入，不能重复启动任务：{path}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def result_lock(directory: Path):
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name(directory.name + ".lock")
    guard = directory.with_name(directory.name + ".writer.lock")
    with _writer_lock(guard):
        owner_path = lock / "owner.json"
        try:
            lock.mkdir()
        except FileExistsError as exc:
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                owner = {}
            if not isinstance(owner, dict) or owner.get("protocol") != PROTOCOL:
                # 老代码不持有操作系统锁，无法判断旧空目录是否仍由活跃进程使用。
                raise RuntimeError(
                    f"发现旧版或未知锁目录：{lock}；确认旧实验进程已退出后，"
                    "仅清理该任务的遗留锁，再执行续跑。不能按目录年龄强行解锁。"
                ) from exc
            # 已拿到 OS 锁，证明采用同协议的原写入者已释放或退出；可接管遗留目录。
        try:
            owner_path.write_text(json.dumps({
                "protocol": PROTOCOL, "pid": os.getpid(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            yield
        finally:
            owner_path.unlink(missing_ok=True)
            lock.rmdir()
