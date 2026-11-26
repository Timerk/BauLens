# Dashboard

Streamlit front end for the BauLens API. It calls the API over HTTP (`BAULENS_API_URL`,
default `http://localhost:8000`) and never imports model code.

```sh
cd apps/dashboard && uv run --locked streamlit run app.py
```

Run it from this directory so Streamlit picks up the theme in `.streamlit/config.toml`.
`BAULENS_API_PUBLIC_URL` sets the API address used for links in the browser when it differs
from `BAULENS_API_URL` (inside Docker Compose).

- `app.py`: page layout, status sidebar and tabs.
- `baulens_dashboard/client.py`: typed HTTP client; tests swap in `httpx.MockTransport`.
- `baulens_dashboard/*_view.py`: concrete mix, crack image and report tabs.
- `examples/`: four CC BY 4.0 images from the Özgenel dataset, see
  [examples/README.md](examples/README.md).

The UI is German; code and comments are English.
