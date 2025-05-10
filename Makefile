setup: sync kernel env-activate

sync:
	uv sync

kernel: sync
	python -m ipykernel install --user --name=new-kernel

env-activate: kernel
	source .venv/bin/activate

