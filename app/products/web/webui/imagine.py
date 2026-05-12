"""WebUI imagine endpoint backed by Grok Imagine WebSocket or chat-based Lite."""

import asyncio
import hmac
import uuid
from typing import Optional

import orjson
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.platform.auth.middleware import get_webui_key, is_webui_enabled
from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_s
from app.products.openai.images import resolve_aspect_ratio

router = APIRouter()

# Models that use chat endpoint (basic tier)
_LITE_MODEL = "grok-imagine-image-lite"
# Models that use WebSocket endpoint (super+ tier)
_WS_MODELS = ("grok-imagine-image", "grok-imagine-image-pro")


async def _acquire_token():
    """Acquire token and model name. Returns (token, account, model_name)."""
    from app.dataplane.account import _directory as _acct_dir
    if _acct_dir is None:
        return None, None, None
    from app.control.model.registry import get as get_model

    # Try imagine models in priority order: super → basic
    for model_name in ("grok-imagine-image", _LITE_MODEL):
        spec = get_model(model_name)
        if spec is None:
            continue
        acct = await _acct_dir.reserve(
            pool_candidates=spec.pool_candidates(),
            mode_id=int(spec.mode_id),
            now_s_override=now_s(),
        )
        if acct is not None:
            return acct.token, acct, model_name

    return None, None, None


def _extract_token(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    scheme, _, token = raw.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return raw


def _is_allowed(token: str) -> bool:
    webui_key = get_webui_key()
    if not webui_key:
        return is_webui_enabled()
    return bool(token) and hmac.compare_digest(token, webui_key)


def _websocket_token(websocket: WebSocket) -> str:
    return (
        _extract_token(websocket.headers.get("authorization"))
        or str(websocket.query_params.get("access_token") or "").strip()
    )


@router.websocket("/imagine/ws")
async def imagine_ws(websocket: WebSocket):
    if not _is_allowed(_websocket_token(websocket)):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    stop_event = asyncio.Event()
    run_task: Optional[asyncio.Task] = None

    async def _send(payload: dict) -> bool:
        try:
            await websocket.send_text(orjson.dumps(payload).decode())
            return True
        except Exception:
            return False

    async def _stop_run():
        nonlocal run_task
        stop_event.set()
        if run_task and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except Exception:
                pass
        run_task = None
        stop_event.clear()

    async def _run(
        prompt: str,
        aspect_ratio: str,
        nsfw: Optional[bool],
        count: int,
        quality: str,
    ):
        from app.dataplane.account import _directory as _acct_dir

        run_id = uuid.uuid4().hex
        enable_pro = quality == "quality"
        await _send({
            "type": "status",
            "status": "running",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "run_id": run_id,
            "count": count,
            "quality": quality,
        })

        acct = None
        try:
            token, acct, model_name = await _acquire_token()
            if not token:
                await _send({
                    "type": "error",
                    "message": "No available accounts for this model tier",
                    "code": "rate_limit_exceeded",
                })
                return

            enable_nsfw = nsfw if nsfw is not None else get_config().get_bool("features.enable_nsfw", True)

            # Route to appropriate backend based on model
            if model_name == _LITE_MODEL:
                # Basic tier: use chat endpoint
                await _run_lite(
                    send_fn=_send,
                    token=token,
                    prompt=prompt,
                    count=count,
                    enable_nsfw=enable_nsfw,
                    run_id=run_id,
                    stop_event=stop_event,
                )
            else:
                # Super+ tier: use WebSocket endpoint
                await _run_ws(
                    send_fn=_send,
                    token=token,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    count=count,
                    enable_nsfw=enable_nsfw,
                    enable_pro=enable_pro,
                    run_id=run_id,
                    stop_event=stop_event,
                )

            if not stop_event.is_set():
                await _send({
                    "type": "status",
                    "status": "completed",
                    "run_id": run_id,
                    "count": count,
                })
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "webui imagine run failed: error_type={} error={}",
                type(exc).__name__,
                exc,
            )
            await _send({
                "type": "error",
                "message": str(exc),
                "code": "internal_error",
            })
        finally:
            if acct and _acct_dir:
                await _acct_dir.release(acct)
            if stop_event.is_set():
                await _send({"type": "status", "status": "stopped", "run_id": run_id})

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except (RuntimeError, WebSocketDisconnect):
                break

            try:
                payload = orjson.loads(raw)
            except Exception:
                await _send({
                    "type": "error",
                    "message": "Invalid message format.",
                    "code": "invalid_payload",
                })
                continue

            action = payload.get("type")
            if action == "start":
                prompt = str(payload.get("prompt") or "").strip()
                if not prompt:
                    await _send({
                        "type": "error",
                        "message": "Prompt cannot be empty.",
                        "code": "invalid_prompt",
                    })
                    continue
                aspect_ratio = resolve_aspect_ratio(str(payload.get("aspect_ratio") or "2:3").strip() or "2:3")
                quality = str(payload.get("quality") or "speed").strip().lower()
                if quality not in {"speed", "quality"}:
                    quality = "speed"
                nsfw = payload.get("nsfw")
                if nsfw is not None:
                    if isinstance(nsfw, str):
                        nsfw = nsfw.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        nsfw = bool(nsfw)
                try:
                    count = int(payload.get("count") or 6)
                except (TypeError, ValueError):
                    count = 6
                count = max(1, min(count, 6))
                await _stop_run()
                run_task = asyncio.create_task(_run(prompt, aspect_ratio, nsfw, count, quality))
                continue

            if action == "stop":
                await _stop_run()
                continue

            await _send({
                "type": "error",
                "message": "Unknown action.",
                "code": "invalid_action",
            })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(
            "webui imagine websocket handler failed: error_type={} error={}",
            type(exc).__name__,
            exc,
        )
    finally:
        await _stop_run()
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1000, reason="Server closing connection")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket-based generation (super+ tier)
# ---------------------------------------------------------------------------

async def _run_ws(
    *,
    send_fn,
    token: str,
    prompt: str,
    aspect_ratio: str,
    count: int,
    enable_nsfw: bool,
    enable_pro: bool,
    run_id: str,
    stop_event: asyncio.Event,
) -> None:
    """Generate images via WebSocket endpoint (super+ tier)."""
    from app.dataplane.reverse.transport.imagine_ws import stream_images

    async for event in stream_images(
        token,
        prompt,
        aspect_ratio=aspect_ratio,
        n=count,
        enable_nsfw=enable_nsfw,
        enable_pro=enable_pro,
    ):
        if stop_event.is_set():
            return
        if not isinstance(event, dict) or event.get("type") == "_meta":
            continue
        event.setdefault("run_id", run_id)
        await send_fn(event)
        if event.get("type") == "error":
            return


# ---------------------------------------------------------------------------
# Chat-based generation (basic tier, lite model)
# ---------------------------------------------------------------------------

async def _run_lite(
    *,
    send_fn,
    token: str,
    prompt: str,
    count: int,
    enable_nsfw: bool,
    run_id: str,
    stop_event: asyncio.Event,
) -> None:
    """Generate images via chat endpoint (basic tier, lite model).

    Uses the same chat-based approach as the /v1/images/generations endpoint
    for the grok-imagine-image-lite model.
    """
    from app.control.model.registry import get as get_model
    from app.products.openai.images import _run_lite_request

    spec = get_model(_LITE_MODEL)
    if spec is None:
        await send_fn({
            "type": "error",
            "message": f"Model {_LITE_MODEL} not found",
            "code": "internal_error",
        })
        return

    cfg = get_config()
    timeout_s = cfg.get_float("chat.timeout", 120.0)

    async def _generate_one(idx: int) -> None:
        if stop_event.is_set():
            return
        try:
            result = await _run_lite_request(
                spec=spec,
                prompt=prompt,
                timeout_s=timeout_s,
                response_format="url",
                progress_cb=None,
            )
            if stop_event.is_set():
                return
            await send_fn({
                "type": "image",
                "image_id": f"{run_id}-{idx}",
                "order": idx,
                "stage": "final",
                "url": result.api_value,
                "blob": "",
                "width": 1024,
                "height": 1024,
                "is_final": True,
                "moderated": False,
                "r_rated": False,
                "run_id": run_id,
            })
        except Exception as exc:
            logger.error("lite image generation failed: idx={} error={}", idx, exc)
            await send_fn({
                "type": "error",
                "message": str(exc),
                "code": "generation_failed",
                "run_id": run_id,
            })

    tasks = [asyncio.create_task(_generate_one(i)) for i in range(count)]
    await asyncio.gather(*tasks, return_exceptions=True)
