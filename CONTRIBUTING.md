# Contributing to BauLens

Start a branch from `main` using `feat/<short-name>`, `fix/<short-name>`, or
`chore/<short-name>`. Keep commits focused and use [Conventional Commits](https://www.conventionalcommits.org/),
for example `feat: verify dataset archives`. Open a pull request to `main`, link the issue it
addresses, and record what you tested. The `lint`, `typecheck`, `test`, and `smoke` checks
must pass before merge.

## Local checks

Run `uv sync --locked` followed by `make all` or the equivalent commands in
[README.md](README.md#development). Run `uv run --locked pre-commit install` to enable the
repository's Git hooks, or run `uv run --locked pre-commit run --all-files` on demand. Use
`uv lock` after an intentional dependency change and commit the updated `uv.lock`.

## Pipeline conventions

- Keep module defaults in `configs/<module>.yaml` and small offline variants in
  `configs/smoke/<module>.yaml`. Add fields to the corresponding Pydantic model in
  `src/baulens/config.py`. Load settings with `load_config(path, Model)` and resolve paths
  relative to the checkout with `project_path(path)`.
- Expose typed `train(config)` and `evaluate(config)` functions from the relevant module
  package. Keep `src/baulens/cli.py` focused on parsing arguments, loading settings, calling
  those functions, and presenting results or errors. Apply the same pattern to report
  generation, model comparison, and ONNX export.
- Read the seed from the run config and call `seed_everything(config.seed)` before a
  training run. Pass the same seed to components that take their own random state.
- Write run outputs under `artifacts/<module>/` and commit selected small figures under
  `docs/figures/<module>/`. Keep raw datasets, model weights, and generated reports local.
- Use `baulens.data.concrete.load_concrete` for the UCI workbook and the attributed CSV
  fixture. The originals for the image dataset and the `tiny_crack_images` pytest fixture
  share the `Positive/` and `Negative/` directory layout.
- Keep smoke tests on CPU. Use fixtures for data and replace HTTP or LLM calls with local
  test doubles. Mark expensive tests with `slow`.
