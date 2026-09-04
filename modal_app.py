"""Modal deployment for the Muse Voice Transcribe x Quail comparison demo.

    modal deploy modal_app.py -e aic-demos

"aic-demos" is a Modal *environment*, not an app name; without -e this would
deploy into Production.

Modal has no built-in per-IP rate limiting, so limits live in app/limits.py.
That limiter keeps per-process state, which is only accurate while a single
container serves every request - hence max_containers=1 below. Raise it only
after moving the limiter to a modal.Dict.
"""

import modal

app = modal.App("muse-transcribe-tyto-vf-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("app", "/root/app")
    .add_local_dir("static", "/root/static")
)

# Create with:
#   modal secret create muse-demo-secrets MODEL_API_KEY=... AIC_SDK_LICENSE=... -e aic-demos
secrets = [modal.Secret.from_name("muse-demo-secrets")]

# Quail and Tyto weights download on first use; a volume keeps them warm.
models = modal.Volume.from_name("muse-demo-models", create_if_missing=True)


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/root/models": models},
    cpu=4,
    memory=8192,
    max_containers=1,      # keeps the in-process rate limiter authoritative
    scaledown_window=300,
    timeout=900,
)
@modal.concurrent(max_inputs=8)  # websockets idle a lot; compare() gates real work
@modal.asgi_app(label="muse-vf")  # URL: ...--muse-vf.modal.run
def web():
    from app.main import app as fastapi_app

    return fastapi_app
