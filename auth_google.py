import os

from authlib.integrations.flask_client import OAuth

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def init_google_oauth(app):
    _load_env_file()
    if certifi is not None:
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    oauth = OAuth(app)

    google_client_id = os.getenv("GOOGLE_CLIENT_ID")
    google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    # If env vars are not set, we still return oauth without registering.
    if not google_client_id or not google_client_secret:
        app.logger.warning(
            "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars."
        )
        return oauth

    # Redirect URI can be configured via GOOGLE_REDIRECT_URI but Authlib can also build it.
    # We'll rely on GOOGLE_REDIRECT_URI for local/dev.
    oauth.register(
        name="google",
        client_id=google_client_id,
        client_secret=google_client_secret,
        access_token_url="https://oauth2.googleapis.com/token",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        api_base_url="https://www.googleapis.com/oauth2/v2/",
        client_kwargs={"scope": "email profile"},
    )

    return oauth

