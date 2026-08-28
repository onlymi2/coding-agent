import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from ke.llm.types import ToolCall


AUTO_APPROVE_TOOLS = frozenset({"read_file", "list_dir", "grep"})
CONFIRM_TOOLS = frozenset({"write_file", "edit_file", "bash"})
MAX_PREVIEW_CHARS = 200


class UnknownPermissionError(LookupError):
    """Raised when a permission id does not belong to this gate."""


class PermissionAlreadyResolvedError(RuntimeError):
    """Raised when a permission has already been answered or cancelled."""


@dataclass(frozen=True)
class PermissionRequest:
    permission_id: str
    tool_call_id: str
    tool_name: str
    preview: str


@dataclass
class _PendingPermission:
    request: PermissionRequest
    allowed: bool | None = None
    resolved: bool = False


class ConfirmationGate(Protocol):
    def requires_confirmation(self, call: ToolCall) -> bool: ...

    def create(self, call: ToolCall) -> PermissionRequest: ...

    def wait(self, permission_id: str) -> bool: ...


def _preview(call: ToolCall) -> str:
    if call.name in {"write_file", "edit_file"}:
        value = call.arguments.get("path", "")
    elif call.name == "bash":
        value = call.arguments.get("command", "")
    else:
        value = call.name
    return str(value)[:MAX_PREVIEW_CHARS]


class PermissionGate:
    """Thread-safe ask/allow gate for local mutating tools."""

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve
        self._condition = threading.Condition()
        self._permissions: dict[str, _PendingPermission] = {}
        self._cancelled = False

    def requires_confirmation(self, call: ToolCall) -> bool:
        return not self.auto_approve and call.name in CONFIRM_TOOLS

    def create(self, call: ToolCall) -> PermissionRequest:
        if not self.requires_confirmation(call):
            raise ValueError(f"工具不需要人工确认：{call.name}")
        request = PermissionRequest(
            permission_id=str(uuid.uuid4()),
            tool_call_id=call.id,
            tool_name=call.name,
            preview=_preview(call),
        )
        with self._condition:
            self._permissions[request.permission_id] = _PendingPermission(
                request=request,
                allowed=False if self._cancelled else None,
                resolved=self._cancelled,
            )
        return request

    def wait(self, permission_id: str) -> bool:
        with self._condition:
            pending = self._permissions.get(permission_id)
            if pending is None:
                raise UnknownPermissionError(permission_id)
            self._condition.wait_for(lambda: pending.resolved)
            return pending.allowed is True

    def resolve(self, permission_id: str, allow: bool) -> PermissionRequest:
        with self._condition:
            pending = self._permissions.get(permission_id)
            if pending is None:
                raise UnknownPermissionError(permission_id)
            if pending.resolved:
                raise PermissionAlreadyResolvedError(permission_id)
            pending.allowed = allow
            pending.resolved = True
            self._condition.notify_all()
            return pending.request

    def cancel_all(self) -> None:
        with self._condition:
            self._cancelled = True
            for pending in self._permissions.values():
                if not pending.resolved:
                    pending.allowed = False
                    pending.resolved = True
            self._condition.notify_all()

    def reset(self) -> None:
        """Allow a new run after the previous run finished or was aborted."""

        with self._condition:
            self._cancelled = False

    def get(self, permission_id: str) -> PermissionRequest | None:
        with self._condition:
            pending = self._permissions.get(permission_id)
            return pending.request if pending is not None else None
